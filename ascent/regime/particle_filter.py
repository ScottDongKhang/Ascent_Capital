"""
ascent/regime/particle_filter.py
---------------------------------
Sequential Monte Carlo (Particle Filter) for online regime probability updates.

Between batch HMM refits, this module updates the posterior distribution over
hidden states daily via importance sampling with resampling (SIR algorithm).

Does NOT refit HMM parameters — only updates state probabilities.

CRITICAL: Must be fully reinitialized when HMM is batch-refit.
State indices may change after refit (K re-labeled). reinitialize() must be
called immediately after every batch refit, passing the new model.

Algorithm (per new observation x_t):
  1. Propagate: each particle's state transitions via HMM transition matrix
  2. Reweight: multiply particle weight by emission likelihood p(x_t | state)
  3. Normalize weights
  4. Resample (SIR): when effective N = 1/Σw² < N/2, resample uniformly
  5. State posterior = weighted histogram over particle states

Integration in engine.py:
  - After every batch refit: pf.reinitialize(new_hmm_model)
  - Each daily run: signal = pf.update(today_observation, state_labels)
  - Batch signal and PF signal should agree within tolerance; divergence > 2σ is logged
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np

log = logging.getLogger(__name__)


class RegimeParticleFilter:
    """
    Online Bayesian regime probability update via Sequential Monte Carlo.

    Parameters
    ----------
    hmm_model : object
        Fitted HMM-like model with attributes:
          - n_components (int): number of hidden states
          - startprob_ (np.ndarray): initial state distribution [K]
          - transmat_ (np.ndarray): transition matrix [K × K]
          - means_ (np.ndarray): Gaussian emission means [K × D]
          - covars_ (np.ndarray): Gaussian emission covariances [K × D] or [K × D × D]
    n_particles : int
        Number of particles (default 500).
    """

    def __init__(self, hmm_model, n_particles: int = 500) -> None:
        self.n_particles = n_particles
        self._init_from_model(hmm_model)

    # ── public API ────────────────────────────────────────────────────────────

    def update(
        self,
        observation: np.ndarray,
        state_labels: Optional[Dict[int, str]] = None,
    ):
        """
        Process one new observation. Returns updated RegimeSignal.

        Call once per trading day after new prices are available.

        Parameters
        ----------
        observation : np.ndarray
            Feature vector for the current day, shape (D,).
        state_labels : dict, optional
            Mapping from state index to label string.
            Example: {0: "calm_bull", 1: "stressed", 2: "crisis"}

        Returns
        -------
        RegimeSignal with label, probability, risk_multiplier.
        """
        self._propagate()
        self._reweight(observation)
        if self._effective_n() < self.n_particles / 2:
            self._resample()
        return self._to_signal(state_labels)

    def reinitialize(self, new_hmm_model) -> None:
        """
        Fully reinitialize after a batch HMM refit.

        MUST be called after every batch refit — do not carry over old particles
        calibrated to the previous model's state space.
        """
        self._init_from_model(new_hmm_model)
        log.info("[PF] Particle filter reinitialized after HMM refit (K=%d)", self.n_states)

    def _effective_n(self) -> float:
        """Effective sample size = 1 / Σw²."""
        return float(1.0 / (np.sum(self.weights ** 2) + 1e-300))

    # ── private helpers ───────────────────────────────────────────────────────

    def _init_from_model(self, hmm_model) -> None:
        """Extract model parameters and initialize particles from stationary distribution."""
        self.hmm = hmm_model
        self.n_states = int(hmm_model.n_components)

        # Extract transition matrix
        self._transmat = np.asarray(hmm_model.transmat_).copy()

        # Extract Gaussian emission parameters
        self._means = np.asarray(hmm_model.means_).copy()        # [K × D]
        covars = np.asarray(hmm_model.covars_).copy()
        # covars can be [K × D] (diag) or [K × D × D] (full)
        if covars.ndim == 2:
            # Diagonal covariance: store as [K × D × D] for consistent scoring
            D = covars.shape[1]
            self._covars = np.array([np.diag(covars[k]) for k in range(self.n_states)])
        elif covars.ndim == 3:
            self._covars = covars
        else:
            D = self._means.shape[1]
            self._covars = np.array([np.eye(D) for _ in range(self.n_states)])

        # Precompute inverse covariances and log-det for Gaussian scoring
        D = self._means.shape[1]
        self._inv_covars = np.zeros_like(self._covars)
        self._log_det_covars = np.zeros(self.n_states)
        for k in range(self.n_states):
            try:
                cov = self._covars[k]
                self._inv_covars[k] = np.linalg.inv(cov + np.eye(D) * 1e-6)
                sign, log_det = np.linalg.slogdet(cov + np.eye(D) * 1e-6)
                self._log_det_covars[k] = log_det if sign > 0 else 0.0
            except np.linalg.LinAlgError:
                self._inv_covars[k] = np.eye(D)
                self._log_det_covars[k] = 0.0

        # Initialize particles from start probability
        start_probs = np.asarray(hmm_model.startprob_)
        start_probs = np.clip(start_probs, 1e-9, None)
        start_probs /= start_probs.sum()
        self.particles = np.random.choice(
            self.n_states, size=self.n_particles, p=start_probs
        )
        self.weights = np.ones(self.n_particles) / self.n_particles

    def _propagate(self) -> None:
        """Propagate each particle forward via transition matrix."""
        new_particles = np.empty(self.n_particles, dtype=int)
        for i, state in enumerate(self.particles):
            trans_probs = self._transmat[state]
            # Clip and renormalize in case of numerical issues
            trans_probs = np.clip(trans_probs, 1e-10, None)
            trans_probs /= trans_probs.sum()
            new_particles[i] = np.random.choice(self.n_states, p=trans_probs)
        self.particles = new_particles

    def _reweight(self, observation: np.ndarray) -> None:
        """Multiply each particle's weight by emission likelihood p(obs | state)."""
        obs = np.asarray(observation, dtype=float)
        D = self._means.shape[1]

        # Align observation dimension to model dimension
        if len(obs) < D:
            obs = np.pad(obs, (0, D - len(obs)))
        elif len(obs) > D:
            obs = obs[:D]

        log_likelihoods = np.zeros(self.n_particles)
        for i, state in enumerate(self.particles):
            diff = obs - self._means[state]
            # Log Gaussian: -0.5 * (diff' @ inv_cov @ diff) - 0.5 * log_det - D/2 * log(2π)
            mahal = float(diff @ self._inv_covars[state] @ diff)
            log_likelihoods[i] = -0.5 * (mahal + self._log_det_covars[state] + D * np.log(2 * np.pi))

        # Numerically stable weight update
        log_weights = np.log(self.weights + 1e-300) + log_likelihoods
        log_weights -= log_weights.max()  # subtract max for stability
        self.weights = np.exp(log_weights)
        total = self.weights.sum()
        if total > 0:
            self.weights /= total
        else:
            self.weights = np.ones(self.n_particles) / self.n_particles

    def _resample(self) -> None:
        """Systematic resampling (SIR) — restores particle diversity."""
        cumsum = np.cumsum(self.weights)
        cumsum[-1] = 1.0  # ensure exact 1.0 at end
        positions = (np.arange(self.n_particles) + np.random.uniform()) / self.n_particles
        new_particles = np.empty(self.n_particles, dtype=int)
        j = 0
        for i in range(self.n_particles):
            while j < len(cumsum) - 1 and cumsum[j] < positions[i]:
                j += 1
            new_particles[i] = self.particles[j]
        self.particles = new_particles
        self.weights = np.ones(self.n_particles) / self.n_particles

    def _to_signal(self, state_labels: Optional[Dict[int, str]] = None):
        """Convert current particle distribution to a RegimeSignal."""
        import pandas as pd
        from ascent.regime.types import RegimeSignal, RegimeLabel, REGIME_CONFIG_DEFAULTS

        # Compute state posterior from weighted particle histogram
        state_probs = np.zeros(self.n_states)
        for i, state in enumerate(self.particles):
            state_probs[state] += self.weights[i]
        state_probs /= (state_probs.sum() + 1e-300)

        dominant_state = int(np.argmax(state_probs))

        # Resolve label string → RegimeLabel enum
        if state_labels is not None:
            label_str = state_labels.get(dominant_state, "uncertain")
        else:
            label_str = "uncertain"

        # Convert string to RegimeLabel enum (graceful fallback)
        try:
            label = RegimeLabel(label_str)
        except ValueError:
            label = RegimeLabel.UNCERTAIN

        # Map label to risk multiplier
        risk_multipliers = REGIME_CONFIG_DEFAULTS.get("regime_risk_multiplier", {})
        risk_mult = float(risk_multipliers.get(label_str, risk_multipliers.get(label.value, 1.0)))

        entropy = float(-np.sum(state_probs * np.log(state_probs + 1e-300)))

        # Default sleeve adjustments (particle filter doesn't tune sleeves)
        sleeve_adj = REGIME_CONFIG_DEFAULTS.get("regime_sleeve_adjustments", {}).get(label_str, {})

        return RegimeSignal(
            date=pd.Timestamp.now().normalize(),
            probs=state_probs,
            label=label,
            entropy=entropy,
            transition_flag=False,
            risk_multiplier=risk_mult,
            sleeve_adjustments=sleeve_adj,
            dwell_days=0,
        )

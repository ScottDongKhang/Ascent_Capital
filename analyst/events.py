"""Named events the one supported question can be asked about.

Hardcoded on purpose. Resolving "the recent conflict in the Middle East" to a
date is a research judgement, and a v0 that guesses at it hides an error the
whole analysis then inherits. Every date here is the market's first full
reaction session or the event day itself, stated per entry.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Event:
    key: str
    name: str
    date: str  # ISO. The session the market first traded the news.
    note: str


EVENTS: dict[str, Event] = {
    e.key: e
    for e in [
        Event(
            key="abqaiq_2019",
            name="Abqaiq drone attack on Saudi processing facility",
            date="2019-09-16",
            note="Attack was Sat 2019-09-14; first trading session is Mon the 16th.",
        ),
        Event(
            key="ukraine_2022",
            name="Russian full-scale invasion of Ukraine",
            date="2022-02-24",
            note="Invasion began pre-market European time on the 24th.",
        ),
        Event(
            key="oct7_2023",
            name="Hamas attack on Israel",
            date="2023-10-09",
            note="Attack was Sat 2023-10-07; first trading session is Mon the 9th.",
        ),
        Event(
            key="iran_israel_2024",
            name="Iranian missile and drone strike on Israel",
            date="2024-04-15",
            note="Strike was Sat 2024-04-13; first trading session is Mon the 15th.",
        ),
    ]
}


def resolve(key: str) -> Event:
    if key not in EVENTS:
        raise KeyError(f"unknown event {key!r}. known: {sorted(EVENTS)}")
    return EVENTS[key]

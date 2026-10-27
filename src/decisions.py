"""Place one action on the next session.

``place(action, as_of, sessions)`` takes an action already equal to
``BUY``, ``SELL``, or ``HOLD``. ``sessions`` is the ordered bar index.
``as_of`` is one of those timestamps. ``execute_on`` is the next
timestamp in that index.

``BUY`` writes the entry boolean on ``execute_on``. ``SELL`` writes the
exit boolean. ``HOLD`` writes neither. The open position is not read.
A ``SELL`` while the sleeve is flat still writes the exit boolean.

``signals(records, sessions)`` stamps each record's booleans on
``execute_on``. A session with no record stays false. This file does
not call vectorbt, and it does not check the boolean it wrote.
"""

import pandas as pd


def place(action, as_of, sessions):
    if action not in ("BUY", "SELL", "HOLD"):
        raise ValueError("action must be BUY, SELL, or HOLD")
    sessions = pd.Index(sessions)
    try:
        loc = int(sessions.get_loc(as_of))
    except (KeyError, TypeError, ValueError):
        raise ValueError("as_of must be one session") from None
    if loc + 1 >= len(sessions):
        raise ValueError("as_of has no next session")
    if action == "BUY":
        entry, exit_ = True, False
    elif action == "SELL":
        entry, exit_ = False, True
    else:
        entry, exit_ = False, False
    return {
        "as_of": sessions[loc],
        "execute_on": sessions[loc + 1],
        "action": action,
        "entry": entry,
        "exit": exit_,
    }


def signals(records, sessions):
    sessions = pd.Index(sessions)
    entries = pd.Series(False, index=sessions)
    exits = pd.Series(False, index=sessions)
    for record in records:
        when = record["execute_on"]
        entries.loc[when] = record["entry"]
        exits.loc[when] = record["exit"]
    return entries, exits

# Vendored JvN29 machine patterns

These are von Neumann's documented machines for the 29-state automaton,
taken unmodified from Golly's standard pattern collection
(`Patterns/Self-Rep/JvN/`, hosted at
https://github.com/AlephAlpha/golly, itself an unofficial mirror of
golly.sourceforge.io). They are redistributed here so the project can run
*real* von Neumann circuitry instead of hand-placed cells.

| File | Rule | Author / source | Purpose |
|:--|:--|:--|:--|
| `counter-demo.rle` | JvN29 | Renato Nobili (`SIMPLE_COUNTERS.EVN`) | Counting circuit: after ten input pulses it emits an output pulse and resets itself. |
| `cell-coders-demo.rle` | JvN29 | Renato Nobili (`Q-STATE_CODERS.JVN`) | The nine signal trains that construct the nine constructible cell types (9–12, 17–20, 25), plus the coders that generate them. |
| `read-arm-demo.rle` | JvN29 | Golly pattern collection | One component of von Neumann's original design: the tape reader. |
| `small-JvN-self-replicator.rle` | JvN29 | Golly pattern collection | A compact self-reproducing machine; it constructs a copy of its own tape-handling machinery. |

Hex/decimal state letters follow the Golly multi-state RLE encoding
implemented in `mesa_self_construction/mstate_rle.py` (`.`/`b` = 0,
`A`–`X` = 1–24, `pA`–`pD` = 25–28). The transition data these run under is
`mesa_self_construction/data/JvN29.table`.

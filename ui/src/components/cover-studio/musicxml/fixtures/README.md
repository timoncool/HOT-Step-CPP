# MusicXML import fixtures

These files exercise browser import against exports from three notation programs. The tests check conversion behavior and ABC syntax; they do not assert that every engraving or playback detail survives.

| File | Source and licence | Export details |
|---|---|---|
| `openscore_nun_seh_ich_wohl.mxl` | [ScoreBase score](https://scorebase.org/scores/252786), marked CC0 (Public Domain) on the source page | MuseScore 4.5.2; complete `.mxl` with a voice and piano part. |
| `kyrie_excerpt.xml` | [Source page](https://drfeezell.com/kyrie-2003-choir-and-piano/), which states the composition was released under CC0 1.0 | Sibelius 23.11.1. Trimmed from the 1.9 MB export to the first eight measures of each of its five parts; all other score metadata is retained. |
| `w3c_tutorial-tablature.musicxml` | [W3C reference](https://www.w3.org/2021/06/musicxml40/musicxml-reference/examples/tutorial-tablature/) and [source file](https://raw.githubusercontent.com/w3c-cg/musicxml/gh-pages/docs/src/data/examples/musicxml/tutorial-tablature.musicxml), under the W3C Community Final Specification Agreement | Finale v28.0 for Mac; pitched and tablature parts. |
| `w3c_tutorial-chord-symbols.musicxml` | [W3C reference](https://www.w3.org/2021/06/musicxml40/musicxml-reference/examples/tutorial-chord-symbols/) and [source file](https://raw.githubusercontent.com/w3c-cg/musicxml/gh-pages/docs/src/data/examples/musicxml/tutorial-chord-symbols.musicxml), under the W3C Community Final Specification Agreement | Finale v28.0 for Mac; harmony symbols on a single staff. |

The W3C source files currently identify Finale v28.0, while the rendered reference pages show an earlier v27.0 export. These fixtures come from the linked source files. Full retrieval and licence evidence is recorded in `RESEARCH/musicxml-fixtures/PROVENANCE.md` in the Buzz workspace.

The six `testsuite_*.musicxml` files come from the [W3C MusicXML test suite](https://github.com/w3c-cg/musicxmlTestSuite/tree/master/xmlFiles). They cover repeat endings, key and meter changes, tempo marks, multiple lyric verses, and transposing instruments. The repository's MIT notice is in `MUSICXML_TEST_SUITE_LICENSE.txt`.

`w3c_score-timewise-element.musicxml` comes from the [W3C MusicXML reference](https://github.com/w3c-cg/musicxml/blob/gh-pages/docs/src/data/examples/musicxml/score-timewise-element.md). The XML declaration and DOCTYPE were added so the embedded example parses as a file. The two `synthetic_*.musicxml` files were written for these tests: one exercises D.C. al Coda and the other combines chord symbols with lyrics. They have no external source.

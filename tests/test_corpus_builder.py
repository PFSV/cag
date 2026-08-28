import tempfile
import unittest
from pathlib import Path

from src.corpus_builder import build_corpus, read_text_files, render_corpus


class CorpusBuilderTests(unittest.TestCase):
    def test_reads_supported_files_in_stable_order(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "b.md").write_text("second", encoding="utf-8")
            (root / "a.txt").write_text("first", encoding="utf-8")
            (root / "ignored.bin").write_bytes(b"not text")

            documents = read_text_files(root)

        self.assertEqual([document["source"] for document in documents], ["a.txt", "b.md"])
        self.assertEqual([document["text"] for document in documents], ["first", "second"])

    def test_renders_sections_and_collapses_extra_blank_lines(self) -> None:
        corpus = render_corpus(
            [
                {"source": "one.txt", "section": "docs", "text": "alpha\n\n\n\nbeta"},
                {"source": "two.txt", "section": "docs", "text": "gamma"},
            ]
        )

        self.assertEqual(corpus, "## docs\n\nalpha\n\nbeta\ngamma")

    def test_build_corpus_writes_a_reusable_text_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw = root / "raw"
            raw.mkdir()
            (raw / "sample.txt").write_text("verified knowledge", encoding="utf-8")
            output = root / "knowledge.txt"

            count = build_corpus(raw, output)

            self.assertEqual(count, 1)
            self.assertEqual(output.read_text(encoding="utf-8"), "## sample.txt\n\nverified knowledge\n")


if __name__ == "__main__":
    unittest.main()

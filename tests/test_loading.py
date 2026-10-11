import tempfile
import unittest
import zipfile
from pathlib import Path

from wingfoil_analyzer.loading import load


class LoadingTest(unittest.TestCase):
    @unittest.skipUnless(
        (Path(__file__).parents[1] / "24277338530_ACTIVITY.fit").exists(),
        "Local activity fixture is not bundled with the public repository",
    )
    def test_loads_bare_fit_and_zip_wrapped_fit(self) -> None:
        fit_path = Path(__file__).parents[1] / "24277338530_ACTIVITY.fit"
        bare = load(fit_path)

        with tempfile.TemporaryDirectory() as directory:
            zip_path = Path(directory) / "activity.zip"
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.write(fit_path, "activity.fit")
            wrapped = load(zip_path)

        self.assertEqual(len(bare[0]), len(wrapped[0]))
        self.assertEqual(bare[1].shape, wrapped[1].shape)
        self.assertEqual(bare[2].shape, wrapped[2].shape)
        self.assertEqual(bare[3].shape, wrapped[3].shape)


if __name__ == "__main__":
    unittest.main()

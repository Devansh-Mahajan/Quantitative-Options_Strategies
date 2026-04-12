import unittest

from core.universe_maintenance import dedupe_symbols, resolve_download_symbol


class UniverseMaintenanceTests(unittest.TestCase):
    def test_dedupe_symbols_normalizes_and_preserves_order(self):
        symbols = [" brk.b ", "AAPL", "aapl", "SQ", "", "BRK.B"]
        self.assertEqual(dedupe_symbols(symbols), ["BRK.B", "AAPL", "SQ"])

    def test_resolve_download_symbol_handles_yahoo_aliases(self):
        self.assertEqual(resolve_download_symbol("BRK.B"), "BRK-B")
        self.assertEqual(resolve_download_symbol("BF.B"), "BF-B")
        self.assertEqual(resolve_download_symbol("SQ"), "XYZ")
        self.assertEqual(resolve_download_symbol("MSFT"), "MSFT")


if __name__ == "__main__":
    unittest.main()

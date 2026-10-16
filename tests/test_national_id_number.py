"""ID parsing and OCR consensus regressions, using synthetic numbers only."""
import unittest

from extractors.national_id import _parse_id_number, _select_id_number


class NationalIdNumberTests(unittest.TestCase):
    def test_arabic_spaced_digits_decode_birth_date(self):
        fields = _parse_id_number("٣ ٠٤ ٠٥٢٥ ١٢ ١٢٣٤٥")
        self.assertEqual(fields["date_of_birth"], "2004-05-25")
        self.assertEqual(fields["governorate"], "الدقهلية")

    def test_invalid_readings_are_not_repaired_into_ids(self):
        for reading in (
            "30413251212345",  # invalid month
            "30402301212345",  # nonexistent date
            "39905251212345",  # future birthday
            "30405259912345",  # unknown governorate
            "304052512123456",  # extra digit
            "304O5251212345",  # letter O is not zero
            "3040525/1212345",  # separate values must not be concatenated
        ):
            with self.subTest(reading=reading):
                self.assertEqual(_parse_id_number(reading), {})

    def test_consensus_normalizes_digit_groups(self):
        self.assertEqual(_select_id_number([
            "٣ ٠٤ ٠٥٢٥ ١٢ ١٢٣٤٥\n", "30405251212345\n", "garbage",
        ]), "30405251212345")

    def test_single_or_tied_readings_are_rejected(self):
        self.assertIsNone(_select_id_number(["30405251212345"]))
        self.assertIsNone(_select_id_number([
            "30405251212345", "30405251212345",
            "30405251212347", "30405251212347",
        ]))

    def test_never_take_substrings_or_join_lines(self):
        for text in ("930405251212345", "3040525\n1212345", "2023/12 30405251212345"):
            with self.subTest(text=text):
                self.assertIsNone(_select_id_number([text, text]))


if __name__ == "__main__":
    unittest.main()

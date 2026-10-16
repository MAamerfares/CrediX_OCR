import unittest

from extractors.bank_statement import _extract_via_regex as parse_bank
from extractors.salary_cert import _extract_via_regex as parse_salary
from main import assemble


class ContractFixTests(unittest.TestCase):
    def test_arabic_net_salary_and_gross_fallback(self):
        self.assertEqual(parse_salary("صافي المرتب: 22,400")['declared_net_salary'], 22400)
        self.assertEqual(parse_salary("الصافي المنصرف: 22,400")['declared_net_salary'], 22400)
        payload = assemble({}, {'declared_gross_salary': 28000}, {}, {})
        self.assertEqual(payload['salary_certificate_fields']['declared_net_salary']['value'], 22400)

    def test_monthly_credits_exclude_debits_and_balances(self):
        text = """Date Description Debit Credit Balance
01/01/2025 Salary 0 28000 4000
05/01/2025 Transfer 0 2000 3500
10/01/2025 Fee 50 0 3450
01/02/2025 Salary 0 28000 4000
10/02/2025 Fee 50 0 3950"""
        parsed = parse_bank(text)
        self.assertEqual(parsed['avg_monthly_net_inflow'], 29000)
        self.assertAlmostEqual(parsed['income_regularity_score'], 0.9512, places=4)

    def test_legacy_score_and_facility_deduplication(self):
        facility = {'lender_name': 'Bank A', 'facility_type': 'Loan',
                    'granted_amount': 950000, 'outstanding_amount': 400000,
                    'start_date': '2025-01-01'}
        payload = assemble({}, {}, {'income_regularity_score': 88.11},
                           {'bureau_facilities': [facility, dict(facility), dict(facility)]})
        self.assertEqual(payload['bank_statement_fields']['income_regularity_score']['value'], 0.8811)
        self.assertEqual(len(payload['iscore_report_fields']['bureau_facilities']), 1)
        self.assertNotIn('fraud_assessment', payload)
        self.assertNotIn('form_data', payload)


if __name__ == '__main__':
    unittest.main()

"""Answer-type compatibility check.

The point of these tests is mostly the NEGATIVE cases: this gate may only fire
on a confident type mismatch. A false abstain deletes a correct answer, which is
worse than the noise it removes, so the "must not abstain" cases outnumber the
"must abstain" ones on purpose.
"""

from __future__ import annotations

import unittest

import _hwiki_path  # noqa: F401,E402

from hwiki import answerability as ab  # noqa: E402


class RequestedTypesTests(unittest.TestCase):
    def test_when_asks_for_a_date(self):
        self.assertEqual(ab.requested_types("when does the Lisbon trip start"), {ab.DATE})

    def test_how_many_asks_for_a_quantity(self):
        self.assertIn(ab.QUANTITY, ab.requested_types("how many runners do we need"))

    def test_salary_asks_for_an_amount(self):
        self.assertEqual(ab.requested_types("what is the on-call salary bonus"), {ab.AMOUNT})

    def test_flight_number_asks_for_an_identifier(self):
        self.assertEqual(
            ab.requested_types("what is the flight number for the Lisbon trip"),
            {ab.IDENTIFIER})

    def test_open_ended_question_asks_for_nothing(self):
        for question in (
            "what is our deployment policy",
            "how should chat replies be formatted",
            "which editor does the user prefer",
            "why do backups run at night",
            "what do we use for backups",
        ):
            self.assertEqual(ab.requested_types(question), set(), question)


class ProvidedTypesTests(unittest.TestCase):
    def test_month_and_year_is_a_date(self):
        self.assertIn(ab.DATE, ab.provided_types("The Lisbon trip now starts in May 2027"))

    def test_weekday_is_a_date(self):
        self.assertIn(ab.DATE, ab.provided_types("The on-call rotation changes every Monday"))

    def test_currency_is_an_amount(self):
        self.assertIn(ab.AMOUNT, ab.provided_types("The seat upgrade costs $40"))

    def test_flight_number_style_id_is_an_identifier(self):
        self.assertIn(ab.IDENTIFIER, ab.provided_types("Booked on BA 2491"))

    def test_a_bare_number_is_not_an_identifier(self):
        # Deliberate: if bare digits counted as identifiers, "May 2027" would
        # satisfy an identifier question and the gate would stop catching the
        # exact mismatch it exists for.
        self.assertNotIn(ab.IDENTIFIER, ab.provided_types("Staging database runs Postgres 15"))
        self.assertNotIn(ab.IDENTIFIER, ab.provided_types("The Lisbon trip starts in May 2027"))

    def test_size_with_unit_is_a_quantity(self):
        self.assertIn(ab.QUANTITY, ab.provided_types("The runner has 8 GB of RAM"))

    def test_plain_prose_has_no_type(self):
        self.assertEqual(ab.provided_types("User prefers short replies with no pings"),
                         set())


class IsAnswerableTests(unittest.TestCase):
    # -- must abstain: the question wants one type, the claim gives another ----

    def test_identifier_question_against_a_date_claim_abstains(self):
        self.assertFalse(ab.is_answerable(
            "what is the flight number for the Lisbon trip",
            "The Lisbon trip now starts in May 2027"))

    def test_amount_question_against_a_schedule_claim_abstains(self):
        self.assertFalse(ab.is_answerable(
            "what is the on-call salary bonus",
            "The on-call rotation changes every Monday"))

    # -- must NOT abstain: correct answers, including the risky shapes --------

    def test_date_question_answered_by_a_date(self):
        self.assertTrue(ab.is_answerable(
            "when does the Lisbon trip start",
            "The Lisbon trip now starts in May 2027"))

    def test_identifier_question_answered_by_an_identifier(self):
        self.assertTrue(ab.is_answerable(
            "what is the postgres version", "Staging database runs Postgres 15"))

    def test_quantity_question_answered_by_a_quantity(self):
        self.assertTrue(ab.is_answerable(
            "how many runners do we need", "CI runs on 4 runners"))

    def test_untyped_answer_is_never_filtered(self):
        # A prose answer can still be the right answer; never guess against it.
        self.assertTrue(ab.is_answerable(
            "what is our deployment policy",
            "No production deploys on Fridays"))

    def test_question_asking_nothing_is_never_filtered(self):
        self.assertTrue(ab.is_answerable(
            "how should chat replies be formatted",
            "User wants short chat replies with no progress pings"))

    def test_evidence_can_supply_the_type(self):
        self.assertTrue(ab.is_answerable(
            "what is the flight number for the Lisbon trip",
            "Flight is booked",
            evidence="Booked on BA 2491, departs in May 2027"))


class FilterHitsTests(unittest.TestCase):
    def test_untyped_query_returns_every_hit(self):
        hits = [{"claim": "a"}, {"claim": "b"}]
        self.assertEqual(ab.filter_hits("how should replies look", hits), hits)

    def test_mismatched_hits_are_dropped(self):
        hits = [{"claim": "The Lisbon trip now starts in May 2027"}]
        self.assertEqual(
            ab.filter_hits("what is the flight number", hits), [])

    def test_kept_hits_survive(self):
        hits = [{"claim": "The Lisbon trip now starts in May 2027"},
                {"claim": "Booked as BA2491"}]
        kept = ab.filter_hits("what is the flight number", hits)
        self.assertEqual([h["claim"] for h in kept], ["Booked as BA2491"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

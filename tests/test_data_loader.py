"""
Unit Tests for DataLoader and Schema Validation Layer.

Covers:
  - Default TSV loading
  - Canonical schema validation (entity_id, business_name, business_address, country)
  - Missing column failure (SchemaValidationError raised, no silent fabrication)
  - Empty values, NaN, and whitespace cleaning
  - Country values handling (unconstrained, including unseen France, Germany, Japan, etc.)
  - Duplicate entity IDs detection and deduplication
  - Source identity preservation (source1, source2, source3)
  - Column ordering invariance
  - Ground truth format parsing (competition & legacy)
  - UTF-8 text encoding preservation
"""

import unittest
import tempfile
from pathlib import Path
import pandas as pd

from src.data.loader import (
    DataLoader,
    SchemaValidationError,
    CANONICAL_SOURCE_COLUMNS,
)


class TestDataLoaderSchema(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)
        self.loader = DataLoader(raw_data_dir=self.data_dir)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_tsv_loading_exact_canonical_columns(self):
        """Tests that standard TSV source files with exact canonical columns load properly."""
        tsv_content = (
            "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
            "e_001\tAcme Corp\t123 Market St, Suite 400\tUS\n"
            "e_002\tTata Consultancy\tGateway Park, Mumbai\tIN\n"
        )
        file_path = self.data_dir / "source1.tsv"
        file_path.write_text(tsv_content, encoding="utf-8")

        df = self.loader.load_source_file(file_path, source_name="source1")

        self.assertEqual(len(df), 2)
        self.assertEqual(list(df.columns[:5]), ["entity_id", "business_name", "business_address", "country", "source"])
        self.assertEqual(df["entity_id"].tolist(), ["e_001", "e_002"])
        self.assertEqual(df["business_name"].tolist(), ["Acme Corp", "Tata Consultancy"])
        self.assertEqual(df["country"].tolist(), ["US", "IN"])
        self.assertEqual(df["source"].tolist(), ["source1", "source1"])

    def test_missing_column_failure_raises_schema_error(self):
        """Tests that missing any required column raises SchemaValidationError without silently fabricating it."""
        # Missing 'business_address' and 'country'
        tsv_content = (
            "entity_id\tbusiness_name\n"
            "e_001\tAcme Corp\n"
        )
        file_path = self.data_dir / "bad_source.tsv"
        file_path.write_text(tsv_content, encoding="utf-8")

        with self.assertRaises(SchemaValidationError) as ctx:
            self.loader.load_source_file(file_path, source_name="source1")

        err_msg = str(ctx.exception)
        self.assertIn("Schema validation failed", err_msg)
        self.assertIn("business_address", err_msg)
        self.assertIn("country", err_msg)

    def test_empty_values_nan_and_whitespace_cleaning(self):
        """Tests that NaNs, empty cells, and leading/trailing whitespaces are cleanly normalized."""
        tsv_content = (
            "  entity_id  \t  business_name  \t  business_address  \t  country  \n"
            "e_001  \t  Beta Global  \t  456 Oak Avenue  \t  US  \n"
            "e_002\t\t\t\n"  # Empty name, address, country
            "e_003\tGamma LLC\t\tFR\n"
        )
        file_path = self.data_dir / "source2.tsv"
        file_path.write_text(tsv_content, encoding="utf-8")

        df = self.loader.load_source_file(file_path, source_name="source2")

        self.assertEqual(len(df), 3)
        self.assertEqual(df.loc[0, "entity_id"], "e_001")
        self.assertEqual(df.loc[0, "business_name"], "Beta Global")
        self.assertEqual(df.loc[0, "business_address"], "456 Oak Avenue")
        self.assertEqual(df.loc[0, "country"], "US")

        # Row with empty values
        self.assertEqual(df.loc[1, "entity_id"], "e_002")
        self.assertEqual(df.loc[1, "business_name"], "")
        self.assertEqual(df.loc[1, "business_address"], "")
        self.assertEqual(df.loc[1, "country"], "")

    def test_unseen_country_values_france_etc(self):
        """Tests that arbitrary and unseen country values (e.g. France, Germany, Japan) are preserved."""
        tsv_content = (
            "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
            "e_fr\tBoulangerie Parisienne\t10 Rue de la Paix, Paris\tFrance\n"
            "e_de\tBMW Group\tPetuelring 130, Munich\tGermany\n"
            "e_jp\tSony Corporation\tMinato City, Tokyo\tJapan\n"
            "e_unknown\tGlobal Traders\tFree Zone\t\n"
        )
        file_path = self.data_dir / "source3.tsv"
        file_path.write_text(tsv_content, encoding="utf-8")

        df = self.loader.load_source_file(file_path, source_name="source3")

        self.assertEqual(len(df), 4)
        self.assertEqual(df["country"].tolist(), ["France", "Germany", "Japan", ""])

    def test_duplicate_entity_ids_handling(self):
        """Tests that duplicate entity IDs are detected, logged, and deduplicated (keeping first)."""
        tsv_content = (
            "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
            "e_001\tAlpha First\tAddress 1\tUS\n"
            "e_001\tAlpha Duplicate\tAddress 1 Dup\tUS\n"
            "e_002\tBeta Unique\tAddress 2\tUK\n"
        )
        file_path = self.data_dir / "dups_source.tsv"
        file_path.write_text(tsv_content, encoding="utf-8")

        df = self.loader.load_source_file(file_path, source_name="source1", deduplicate=True)

        # Should deduplicate e_001, keeping first occurrence
        self.assertEqual(len(df), 2)
        self.assertEqual(df["entity_id"].tolist(), ["e_001", "e_002"])
        self.assertEqual(df.loc[0, "business_name"], "Alpha First")

    def test_preservation_of_source_identity(self):
        """Tests that Source 1, 2, and 3 each retain their source identity."""
        s1_file = self.data_dir / "source1.tsv"
        s2_file = self.data_dir / "source2.tsv"
        s3_file = self.data_dir / "source3.tsv"

        s1_file.write_text("entity_id\tbusiness_name\tbusiness_address\tcountry\ns1_1\tName1\tAddr1\tUS\n", encoding="utf-8")
        s2_file.write_text("entity_id\tbusiness_name\tbusiness_address\tcountry\ns2_1\tName2\tAddr2\tUS\n", encoding="utf-8")
        s3_file.write_text("entity_id\tbusiness_name\tbusiness_address\tcountry\ns3_1\tName3\tAddr3\tUS\n", encoding="utf-8")

        s1_df = self.loader.load_source_file(s1_file, source_name="source1")
        s2_df = self.loader.load_source_file(s2_file, source_name="source2")
        s3_df = self.loader.load_source_file(s3_file, source_name="source3")

        self.assertEqual(s1_df.loc[0, "source"], "source1")
        self.assertEqual(s2_df.loc[0, "source"], "source2")
        self.assertEqual(s3_df.loc[0, "source"], "source3")

    def test_unexpected_column_ordering(self):
        """Tests that columns given in unexpected order are properly identified and reorganized."""
        tsv_content = (
            "country\tbusiness_address\tentity_id\tbusiness_name\n"
            "France\t12 Boulevard Saint-Germain\ts1_99\tCafé de Flore\n"
        )
        file_path = self.data_dir / "scrambled_source.tsv"
        file_path.write_text(tsv_content, encoding="utf-8")

        df = self.loader.load_source_file(file_path, source_name="source1")

        self.assertEqual(list(df.columns[:5]), ["entity_id", "business_name", "business_address", "country", "source"])
        self.assertEqual(df.loc[0, "entity_id"], "s1_99")
        self.assertEqual(df.loc[0, "business_name"], "Café de Flore")
        self.assertEqual(df.loc[0, "country"], "France")

    def test_competition_ground_truth_tsv_loading(self):
        """Tests loading competition ground truth TSV format (source1_entity_id, matched_entity_ids)."""
        gt_content = (
            "source1_entity_id\tmatched_entity_ids\n"
            "s1_001\ts2_101,s3_201\n"
            "s1_002\ts2_102\n"
            "s1_003\t\n"
        )
        gt_file = self.data_dir / "train_matches.tsv"
        gt_file.write_text(gt_content, encoding="utf-8")

        gt_df = self.loader.load_train_matches(gt_file)

        self.assertEqual(len(gt_df), 3)
        self.assertEqual(gt_df["source1_entity_id"].tolist(), ["s1_001", "s1_002", "s1_003"])
        self.assertEqual(gt_df["matched_entity_ids"].tolist(), ["s2_101,s3_201", "s2_102", ""])

    def test_utf8_encoding_preservation(self):
        """Tests that UTF-8 international characters are accurately loaded."""
        tsv_content = (
            "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
            "s1_utf\tMünchener Rückversicherungs-Gesellschaft\tKöniginstraße 107, München\tGermany\n"
            "s1_jp\tトヨタ自動車株式会社\t愛知県豊田市トヨタ町1番地\tJapan\n"
        )
        file_path = self.data_dir / "utf8_source.tsv"
        file_path.write_text(tsv_content, encoding="utf-8")

        df = self.loader.load_source_file(file_path, source_name="source1")

        self.assertEqual(len(df), 2)
        self.assertIn("Münchener", df.loc[0, "business_name"])
        self.assertIn("トヨタ", df.loc[1, "business_name"])


if __name__ == "__main__":
    unittest.main()

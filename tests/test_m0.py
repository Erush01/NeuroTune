import json
import tempfile
import unittest
from pathlib import Path

from src.data.join_audit import build_audit
from src.data.parse_lastfm import parse_lastfm
from src.data.parse_msd import parse_msd
from src.data.parse_mxm import parse_mxm
from src.data.parse_tagtraum import parse_tagtraum


class ParserTests(unittest.TestCase):
    def test_msd_parser_normalizes_ids_and_counts_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "msd.csv"
            path.write_text(
                "track_id,tempo,artist_id,song_id,artist_name,title\n"
                " trabc ,120,ar1,so1,Artist,Song\n"
                "TRABC,120,ar1,so1,Artist,Song\n",
                encoding="utf-8",
            )
            tracks = parse_msd(path)
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0]["track_id"], "TRABC")
        self.assertEqual(tracks[0]["duplicate_rows"], 1)

    def test_mxm_parser_aggregates_sparse_word_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mxm.csv"
            path.write_text(
                "track_id,word_id,count\nTRABC,7,2\nTRABC,7,3\nTRABC,8,1\n",
                encoding="utf-8",
            )
            songs = parse_mxm(path)
        self.assertEqual(songs["TRABC"], {"7": 5, "8": 1})

    def test_mxm_parser_reads_official_sparse_text_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mxm.txt"
            path.write_text(
                "# comment\n%hello,world\nTRABC,12345,1:2,2:4\n",
                encoding="utf-8",
            )
            songs = parse_mxm(path)
        self.assertEqual(songs, {"TRABC": {"1": 2, "2": 4}})

    def test_tagtraum_parser_reads_primary_genre(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "genres.csv"
            path.write_text("track_id,genre\nTRABC,Rock\n", encoding="utf-8")
            genres = parse_tagtraum(path)
        self.assertEqual(genres, {"TRABC": "Rock"})

    def test_lastfm_parser_reads_scored_edges(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lastfm.tsv"
            path.write_text("track_id\tsimilar_track_id\tscore\nTRABC\tTRDEF\t0.8\n", encoding="utf-8")
            edges = parse_lastfm(path)
        self.assertEqual(edges, [{"source": "TRABC", "target": "TRDEF", "score": 0.8}])

    def test_lastfm_parser_reads_official_per_track_json(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "TRABC.json").write_text(
                json.dumps({"track_id": "TRABC", "similars": [["TRDEF", 0.75]]}),
                encoding="utf-8",
            )
            (root / "TRNOP.json").write_text(
                json.dumps({"track_id": "TRNOP", "tags": []}), encoding="utf-8"
            )
            edges = parse_lastfm(root)
        self.assertEqual(edges, [{"source": "TRABC", "target": "TRDEF", "score": 0.75}])


class AuditTests(unittest.TestCase):
    def test_audit_reports_intersections_retained_edges_and_artist_disjoint_splits(self):
        tracks = [
            {"track_id": f"TR{i}", "tempo": 100.0, "artist_id": f"AR{i}", "song_id": f"SO{i}", "artist_name": f"Artist {i}", "title": f"Song {i}", "duplicate_rows": 0}
            for i in range(20)
        ]
        lyrics = {f"TR{i}": {"1": 1} for i in range(20)}
        genres = {f"TR{i}": "Rock" for i in range(20)}
        edges = [{"source": f"TR{i}", "target": f"TR{i + 1}", "score": 1.0} for i in range(19)]

        audit = build_audit(tracks, lyrics, genres, edges, seed=23)

        self.assertEqual(audit["intersections"]["msd_and_mxm"], 20)
        self.assertEqual(audit["intersections"]["msd_mxm_cd2c"], 20)
        self.assertEqual(audit["intersections"]["covered_by_lastfm"], 20)
        self.assertEqual(audit["positive_edges"]["retained_full_feature_edges"], 19)
        parts = audit["splits"]["artist_disjoint"]
        memberships = {track_id: part for part, ids in parts["track_ids"].items() for track_id in ids}
        self.assertEqual(len(memberships), 20)
        split_edges = parts["usable_positive_edges"]
        self.assertLessEqual(sum(split_edges.values()), 19)
        self.assertEqual(
            audit["feasibility_review"]["test_positive_edges"],
            split_edges["test"],
        )
        self.assertIsNone(audit["feasibility_review"]["minimum_test_edges_predeclared"])
        self.assertEqual(
            sum(split_edges.values()) + parts["cross_partition_edges_excluded"],
            19,
        )

    def test_audit_keeps_one_deterministic_track_per_artist_song_pair(self):
        tracks = [
            {"track_id": "TRZ", "tempo": 100.0, "artist_id": "AR1", "song_id": "SO1", "artist_name": "Artist", "title": "Song", "duplicate_rows": 0},
            {"track_id": "TRA", "tempo": 100.0, "artist_id": "AR1", "song_id": "SO1", "artist_name": "Artist", "title": "Song", "duplicate_rows": 0},
        ]
        audit = build_audit(
            tracks,
            {"TRA": {"1": 1}, "TRZ": {"1": 1}},
            {"TRA": "Rock", "TRZ": "Rock"},
            [{"source": "TRA", "target": "TRZ", "score": 1.0}],
        )
        self.assertEqual(audit["intersections"]["full_feature_tracks"], 1)
        self.assertEqual(audit["duplicates"]["artist_song_duplicate_tracks_excluded"], 1)

    def test_audit_excludes_official_duplicate_group_releases(self):
        tracks = [
            {"track_id": f"TR{i}", "tempo": 100.0, "artist_id": f"AR{i}", "song_id": f"SO{i}", "artist_name": f"Artist {i}", "title": f"Song {i}", "duplicate_rows": 0}
            for i in range(20)
        ]
        audit = build_audit(
            tracks,
            {track["track_id"]: {"1": 1} for track in tracks},
            {track["track_id"]: "Rock" for track in tracks},
            [{"source": "TR0", "target": "TR1", "score": 1.0}],
            duplicate_groups=[["TR0", "TR1"]],
        )
        self.assertEqual(audit["duplicates"]["official_duplicate_tracks_excluded"], 1)
        self.assertEqual(audit["intersections"]["full_feature_tracks"], 19)
        self.assertEqual(audit["positive_edges"]["retained_full_feature_edges"], 0)

    def test_cli_missing_data_output_marks_m0_blocked_and_counts_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "results" / "data_audit.json"
            from scripts.check_data import run

            run(root / "data", output)
            payload = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "blocked_missing_sources")
        self.assertIsNone(payload["source_counts"]["msd"])
        self.assertEqual(len(payload["missing_sources"]), 4)


if __name__ == "__main__":
    unittest.main()

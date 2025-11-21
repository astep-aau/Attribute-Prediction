import pytest
import pandas as pd
from pathlib import Path
from src.data_manipulation.file_loader import FileLoader

# Fixture data
@pytest.fixture
def tmp_csv_files(tmp_path):
    """Create temporary CSV files for testing"""
    # Day 1 data - has edge0, edge1, edge2
    day1 = tmp_path / "day1.csv"
    day1_data = """time_slot,edge0_traversal_time_sec,edge1_traversal_time_sec,edge2_traversal_time_sec
00:05,10,20,30
00:10,15,25,35
00:15,20,30,40"""
    day1.write_text(day1_data)

    # Day 2 data - has edge0, edge2, edge3
    day2 = tmp_path / "day2.csv"
    day2_data = """time_slot,edge0_traversal_time_sec,edge2_traversal_time_sec,edge3_traversal_time_sec
00:05,12,22,32
00:10,17,27,37"""
    day2.write_text(day2_data)

    # Day 3 data - has edge1
    day3 = tmp_path / "day3.csv"
    day3_data = """time_slot,edge1_traversal_time_sec
00:05,100
00:10,110"""
    day3.write_text(day3_data)

    return [str(day1), str(day2), str(day3)]

@pytest.fixture
def tmp_meta_data(tmp_path):
    """Create temporary metadata JSON file"""
    meta_file = tmp_path / "meta.json"
    meta_data = """{
        "1": {"oneway": true, "road_type": "highway", "nodes": [1, 2]},
        "2": {"oneway": false, "road_type": "street", "nodes": [2, 3]},
        "3": {"oneway": true, "road_type": "avenue", "nodes": [3, 4]}
    }"""
    meta_file.write_text(meta_data)
    return str(meta_file)

@pytest.fixture
def tmp_adjacency(tmp_path):
    """Create temporary adjacency CSV file"""
    adj_file = tmp_path / "adjacency.csv"
    adj_data = """edge_id,vertex_start_id,vertex_end_id
0,1,2
1,2,3
2,3,4"""
    adj_file.write_text(adj_data)
    return str(adj_file)

@pytest.fixture
def file_loader(tmp_csv_files, tmp_adjacency, tmp_meta_data):
    """Create FileLoader instance with test data"""
    edge_paths = ",".join(tmp_csv_files)
    return FileLoader(edge_paths, tmp_adjacency, tmp_meta_data)

class TestGetTravelData:

    def test_combines_multiple_files(self, file_loader):
        """Test that all CSV files are combined"""
        result = file_loader.get_travel_data()

        # Should have data from all 3 days
        assert result['day'].nunique() == 3
        assert 1 in result['day'].values
        assert 2 in result['day'].values
        assert 3 in result['day'].values

    def test_includes_all_edges(self, file_loader):
        """Test that all edges from all files are included"""
        result = file_loader.get_travel_data()

        # Should have all edges: edge1, edge2, edge3, edge4
        expected_edges = ['edge0_traversal_time_sec', 'edge1_traversal_time_sec', 'edge2_traversal_time_sec', 'edge3_traversal_time_sec']
        for edge in expected_edges:
            assert edge in result.columns, f"Missing edge: {edge}"

    def test_missing_edges_filled_with_minus_one(self, file_loader):
        """Test that missing edges are filled with -1"""
        result = file_loader.get_travel_data()

        # Day 1 doesn't have edge3, should be -1
        day1_data = result[result['day'] == 1]
        assert (day1_data['edge3_traversal_time_sec'] == -1).all()

        # Day 2 doesn't have edge1, should be -1
        day2_data = result[result['day'] == 2]
        assert (day2_data['edge1_traversal_time_sec'] == -1).all()

        # Day 3 only has edge1, all others should be -1
        day3_data = result[result['day'] == 3]
        assert (day3_data['edge0_traversal_time_sec'] == -1).all()
        assert (day3_data['edge2_traversal_time_sec'] == -1).all()
        assert (day3_data['edge3_traversal_time_sec'] == -1).all()

    def test_edges_sorted_correctly(self, file_loader):
        """Test that edges are sorted by numeric ID"""
        result = file_loader.get_travel_data()

        edge_cols = [col for col in result.columns if col.startswith('edge')]
        # Should be ordered: edge0, edge1, edge2, edge3
        assert edge_cols == ['edge0_traversal_time_sec', 'edge1_traversal_time_sec', 'edge2_traversal_time_sec', 'edge3_traversal_time_sec']

    def test_time_slot_column_first(self, file_loader):
        """Test that time_slot is the first column"""
        result = file_loader.get_travel_data()
        assert result.columns[0] == 'time_slot'

    def test_day_column_added(self, file_loader):
        """Test that day column is added and correct"""
        result = file_loader.get_travel_data()

        assert 'day' in result.columns
        assert result['day'].min() == 1
        assert result['day'].max() == 3

    def test_correct_number_of_rows(self, file_loader):
        """Test total row count matches sum of all files"""
        result = file_loader.get_travel_data()

        # Day 1: 3 rows, Day 2: 2 rows, Day 3: 2 rows = 7 total
        assert len(result) == 7

    def test_preserves_original_values(self, file_loader):
        """Test that original values are preserved"""
        result = file_loader.get_travel_data()








        # Check day 1, time_slot 00:05, edge0 should be 10
        day1_slot0 = result[(result['day'] == 1) & (result['time_slot'] == '00:05')]
        assert day1_slot0['edge0_traversal_time_sec'].values[0] == 10

        # Check day 2, time_slot 00:10, edge2 should be 27
        day2_slot1 = result[(result['day'] == 2) & (result['time_slot'] == '00:10')]
        assert day2_slot1['edge2_traversal_time_sec'].values[0] == 27

# Tests for get_meta_data()
class TestGetMetaData:

    def test_returns_transposed_dataframe(self, file_loader):
        """Test that metadata is transposed (edges as index)"""
        result = file_loader.get_meta_data()

        # Index should be edge IDs
        assert 1 in result.index
        assert 2 in result.index
        assert 3 in result.index

    def test_has_correct_columns(self, file_loader):
        """Test that metadata has expected columns"""
        result = file_loader.get_meta_data()

        expected_cols = ['oneway', 'road_type', 'nodes']
        for col in expected_cols:
            assert col in result.columns

    def test_metadata_values_correct(self, file_loader):
        """Test that metadata values are correctly loaded"""
        result = file_loader.get_meta_data()

        # Check 1 metadata
        assert result.loc[1, 'oneway'] == True
        assert result.loc[1, 'road_type'] == 'highway'
        assert result.loc[1, 'nodes'] == [1, 2]

        # Check 2 metadata
        assert result.loc[2, 'oneway'] == False
        assert result.loc[2, 'road_type'] == 'street'

# Tests for get_adjacency()
class TestGetAdjacency:

    def test_returns_dataframe(self, file_loader):
        """Test that adjacency returns a DataFrame"""
        result = file_loader.get_adjacency()
        assert isinstance(result, pd.DataFrame)

    def test_has_correct_columns(self, file_loader):
        """Test adjacency has expected columns"""
        result = file_loader.get_adjacency()

        expected_cols = ['edge_id', 'vertex_start_id', 'vertex_end_id']
        for col in expected_cols:
            assert col in result.columns

    def test_correct_number_of_edges(self, file_loader):
        """Test adjacency has correct number of connections"""
        result = file_loader.get_adjacency()
        assert len(result) == 3

    def test_adjacency_values_correct(self, file_loader):
        """Test adjacency values are loaded correctly"""
        result = file_loader.get_adjacency()

        # Check first connection
        first_row = result.iloc[0]
        assert first_row['edge_id'] == 0
        assert first_row['vertex_start_id'] == 1
        assert first_row['vertex_end_id'] == 2

# Edge cases and error handling
class TestEdgeCases:

    def test_single_file(self, tmp_path, tmp_adjacency, tmp_meta_data):
        """Test with only one CSV file"""
        single_file = tmp_path / "single.csv"
        single_file.write_text("time_slot,edge1_0\n0,100\n1,200")

        loader = FileLoader(str(single_file), tmp_adjacency, tmp_meta_data)
        result = loader.get_travel_data()

        assert len(result) == 2
        assert result['day'].nunique() == 1

    def test_empty_csv(self, tmp_path, tmp_adjacency, tmp_meta_data):
        """Test handling of empty CSV (only headers)"""
        empty_file = tmp_path / "empty.csv"
        empty_file.write_text("time_slot,edge1_0\n")

        loader = FileLoader(str(empty_file), tmp_adjacency, tmp_meta_data)
        result = loader.get_travel_data()

        assert len(result) == 0

    def test_edge_sorting_with_large_numbers(self, tmp_path, tmp_adjacency, tmp_meta_data):
        """Test edge sorting works with multi-digit edge numbers"""
        file1 = tmp_path / "test.csv"
        file1.write_text("time_slot,edge0_traversal_time_sec,edge132142_traversal_time_sec,edge9999999_traversal_time_sec\n0,1,2,3")

        loader = FileLoader(str(file1), tmp_adjacency, tmp_meta_data)
        result = loader.get_travel_data()

        edge_cols = [col for col in result.columns if col.startswith('edge')]
        # Should be: 0, 132142, 9999999 (numeric sort, not alphabetic)
        assert edge_cols == ['edge0_traversal_time_sec', 'edge132142_traversal_time_sec', 'edge9999999_traversal_time_sec']

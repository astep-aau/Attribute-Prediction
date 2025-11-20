import pytest
import torch
from torch_geometric.data import Data
from src.data_manipulation.file_loader import FileLoader
from src.data_manipulation.graph_dataset_builder import GraphDatasetBuilder


# Fixture data
@pytest.fixture
def tmp_csv_files(tmp_path):
    """Create temporary CSV files with travel data - larger dataset with 10 edges, 3 days, 12 timesteps each"""
    # Day 1 data - all 10 edges, 12 timesteps (hourly from 00:00 to 11:00)
    day1 = tmp_path / "day1.csv"
    day1_data = """time_slot,edge0_traversal_time_sec,edge1_traversal_time_sec,edge2_traversal_time_sec,edge3_traversal_time_sec,edge4_traversal_time_sec,edge5_traversal_time_sec,edge6_traversal_time_sec,edge7_traversal_time_sec,edge8_traversal_time_sec,edge9_traversal_time_sec
00:00,120,85,95,110,75,130,90,105,80,125
01:00,115,80,90,105,70,125,85,100,75,120
02:00,110,75,85,100,65,120,80,95,70,115
03:00,105,70,80,95,60,115,75,90,65,110
04:00,100,65,75,90,55,110,70,85,60,105
05:00,95,60,70,85,50,105,65,80,55,100
06:00,130,95,105,120,85,140,100,115,90,135
07:00,150,110,120,140,100,160,115,130,105,155
08:00,180,140,150,170,130,190,145,160,135,185
09:00,170,130,140,160,120,180,135,150,125,175
10:00,160,120,130,150,110,170,125,140,115,165
11:00,150,110,120,140,100,160,115,130,105,155"""
    day1.write_text(day1_data)

    # Day 2 data - missing edges 3 and 7, 12 timesteps
    day2 = tmp_path / "day2.csv"
    day2_data = """time_slot,edge0_traversal_time_sec,edge1_traversal_time_sec,edge2_traversal_time_sec,edge4_traversal_time_sec,edge5_traversal_time_sec,edge6_traversal_time_sec,edge8_traversal_time_sec,edge9_traversal_time_sec
00:00,125,90,100,80,135,95,85,130
01:00,120,85,95,75,130,90,80,125
02:00,115,80,90,70,125,85,75,120
03:00,110,75,85,65,120,80,70,115
04:00,105,70,80,60,115,75,65,110
05:00,100,65,75,55,110,70,60,105
06:00,135,100,110,90,145,105,95,140
07:00,155,115,125,105,165,120,110,160
08:00,185,145,155,135,195,150,140,190
09:00,175,135,145,125,185,140,130,180
10:00,165,125,135,115,175,130,120,170
11:00,155,115,125,105,165,120,110,160"""
    day2.write_text(day2_data)

    # Day 3 data - only edges 0, 2, 4, 6, 8 (even edges), 12 timesteps
    day3 = tmp_path / "day3.csv"
    day3_data = """time_slot,edge0_traversal_time_sec,edge2_traversal_time_sec,edge4_traversal_time_sec,edge6_traversal_time_sec,edge8_traversal_time_sec
00:00,118,93,73,88,78
01:00,113,88,68,83,73
02:00,108,83,63,78,68
03:00,103,78,58,73,63
04:00,98,73,53,68,58
05:00,93,68,48,63,53
06:00,128,103,83,98,88
07:00,148,118,98,113,103
08:00,178,148,128,143,133
09:00,168,138,118,133,123
10:00,158,128,108,123,113
11:00,148,118,98,113,103"""
    day3.write_text(day3_data)

    return [str(day1), str(day2), str(day3)]


@pytest.fixture
def tmp_meta_data(tmp_path):
    """Create temporary metadata JSON file with road types and oneway info - 10 ways covering all vertices"""
    meta_file = tmp_path / "meta.json"
    # Ways that contain the vertices used in our edges (1-15)
    meta_data = """{
        "1": {"oneway": true, "road_type": "primary", "nodes": ["1", "2", "3"]},
        "2": {"oneway": false, "road_type": "secondary", "nodes": ["3", "4", "5"]},
        "3": {"oneway": true, "road_type": "residential", "nodes": ["2", "6"]},
        "4": {"oneway": false, "road_type": "tertiary", "nodes": ["5", "7", "8"]},
        "5": {"oneway": true, "road_type": "motorway", "nodes": ["8", "9"]},
        "6": {"oneway": false, "road_type": "trunk", "nodes": ["9", "10", "11"]},
        "7": {"oneway": true, "road_type": "unclassified", "nodes": ["11", "12"]},
        "8": {"oneway": false, "road_type": "service", "nodes": ["12", "13", "14"]},
        "9": {"oneway": true, "road_type": "motorway_link", "nodes": ["14", "15"]},
        "10": {"oneway": false, "road_type": "primary_link", "nodes": ["6", "10", "13"]}
    }"""
    meta_file.write_text(meta_data)
    return str(meta_file)


@pytest.fixture
def tmp_adjacency(tmp_path):
    """Create temporary adjacency CSV file representing road network connectivity - 10 edges forming a complex network"""
    adj_file = tmp_path / "adjacency.csv"
    # Network structure (more realistic with branches and convergence):
    #        ┌─→ 4 ──→ 5 ─┐
    #        │   (3)  (4)  │
    #   1 ──→ 2 ──→ 3      ├──→ 7 ──→ 8 ──→ 9
    #  (0)   │  (1) (2)    │   (6)   (7)   (8)
    #        │             │
    #        └─→ 6 ────────┘
    #           (5)
    #
    #   9 ──→ 10 (edge 9)
    #  (8)   (9)
    #
    # Multiple paths from 2 to 7 (via 3→4→5 or via 6)
    # Edges can converge and diverge, creating complex line graph patterns
    adj_data = """edge_id,vertex_start_id,vertex_end_id
0,1,2
1,2,3
2,3,4
3,4,5
4,5,7
5,2,6
6,6,7
7,7,8
8,8,9
9,9,10"""
    adj_file.write_text(adj_data)
    return str(adj_file)


@pytest.fixture
def file_loader(tmp_csv_files, tmp_adjacency, tmp_meta_data):
    """Create FileLoader instance with test data"""
    edge_paths = ",".join(tmp_csv_files)
    return FileLoader(edge_paths, tmp_adjacency, tmp_meta_data)


@pytest.fixture
def graph_builder(file_loader):
    """Create GraphDatasetBuilder instance with default timestep"""
    return GraphDatasetBuilder(file_loader, timestep=0)


class TestInitialization:
    """Test initialization and basic setup of GraphDatasetBuilder"""

    def test_creates_instance(self, file_loader):
        """Test that GraphDatasetBuilder can be instantiated"""
        builder = GraphDatasetBuilder(file_loader, timestep=0)
        assert builder is not None

    def test_edge_as_node_map_created(self, graph_builder):
        """Test that edge-to-node mapping is created"""
        assert len(graph_builder._edge_as_node_map) > 0
        assert isinstance(graph_builder._edge_as_node_map, dict)

    def test_filters_adjacency_to_travel_edges(self, graph_builder):
        """Test that only edges with travel data are included"""
        # All edges 0-9 should be present since they have travel data
        assert len(graph_builder._adjacency_df) == 10
        for i in range(10):
            assert i in graph_builder._edge_as_node_map


class TestLineGraphEdgeIndex:
    """Test line graph connectivity construction"""

    def test_edge_index_is_tensor(self, graph_builder):
        """Test that edge_index is a PyTorch tensor"""
        assert isinstance(graph_builder.edge_index, torch.Tensor)

    def test_edge_index_shape(self, graph_builder):
        """Test that edge_index has shape [2, num_edges]"""
        assert graph_builder.edge_index.dim() == 2
        assert graph_builder.edge_index.size(0) == 2

    def test_edge_index_dtype(self, graph_builder):
        """Test that edge_index has correct dtype"""
        assert graph_builder.edge_index.dtype == torch.long

    def test_creates_correct_connectivity(self, graph_builder):
        """Test that line graph edges are created correctly in a complex network"""
        # Network has branches and convergence:
        # - Edge 0 (1->2) connects to edges 1 and 5 (both start at vertex 2) - BRANCH
        # - Edge 1 (2->3) connects to edge 2 (3->4)
        # - Edge 2 (3->4) connects to edge 3 (4->5)
        # - Edge 3 (4->5) connects to edge 4 (5->7)
        # - Edge 4 (5->7) connects to edge 7 (7->8)
        # - Edge 5 (2->6) connects to edge 6 (6->7)
        # - Edge 6 (6->7) connects to edge 7 (7->8)
        # - Edge 7 (7->8) connects to edge 8 (8->9)
        # - Edge 8 (8->9) connects to edge 9 (9->10)
        # Note: Edges 4 and 6 both feed into edge 7 - CONVERGENCE
        edge_index = graph_builder.edge_index
        edges_list = edge_index.t().tolist()

        # Expected connections (multiple paths due to branching)
        expected_connections = [
            [0, 1],  # 1->2 to 2->3
            [0, 5],  # 1->2 to 2->6 (branch)
            [1, 2],  # 2->3 to 3->4
            [2, 3],  # 3->4 to 4->5
            [3, 4],  # 4->5 to 5->7
            [4, 7],  # 5->7 to 7->8
            [5, 6],  # 2->6 to 6->7
            [6, 7],  # 6->7 to 7->8 (convergence point)
            [7, 8],  # 7->8 to 8->9
            [8, 9],  # 8->9 to 9->10
        ]

        # Verify we have the expected number of connections
        assert edge_index.size(1) == len(expected_connections)

        # Check all expected connections exist
        for conn in expected_connections:
            assert conn in edges_list, f"Missing connection: {conn}"

    def test_branching_from_single_edge(self, graph_builder):
        """Test that edges can branch to multiple successor edges"""
        edge_index = graph_builder.edge_index
        edges_list = edge_index.t().tolist()

        # Edge 0 (1->2) should branch to both edge 1 (2->3) and edge 5 (2->6)
        edge_0_successors = [dest for src, dest in edges_list if src == 0]
        assert len(edge_0_successors) == 2
        assert 1 in edge_0_successors
        assert 5 in edge_0_successors

        # Edge 4 (5->7) should branch to edge 6 (6->7) and edge 7 (7->8)
        edge_4_successors = [dest for src, dest in edges_list if src == 4]
        assert len(edge_4_successors) >= 1  # At least connects to edge 7

    def test_convergence_to_single_edge(self, graph_builder):
        """Test that multiple edges can converge to a single successor edge"""
        edge_index = graph_builder.edge_index
        edges_list = edge_index.t().tolist()

        # Edge 7 (7->8) should have multiple predecessors:
        # - Edge 4 (5->7)
        # - Edge 6 (6->7)
        edge_7_predecessors = [src for src, dest in edges_list if dest == 7]
        assert len(edge_7_predecessors) == 2
        assert 4 in edge_7_predecessors
        assert 6 in edge_7_predecessors

    def test_directed_connections(self, tmp_path):
        """Test that connections are directional (A->B doesn't imply B->A)"""
        # Create a scenario with one-way connection
        csv = tmp_path / "travel.csv"
        csv.write_text("time_slot,edge0_traversal_time_sec,edge1_traversal_time_sec\n00:05,10,20")

        adj = tmp_path / "adj.csv"
        # Edge 0: 1->2, Edge 1: 3->2 (both end at 2, but no connection)
        adj.write_text("edge_id,vertex_start_id,vertex_end_id\n0,1,2\n1,3,2")

        meta = tmp_path / "meta.json"
        meta.write_text('{"1": {"oneway": true, "road_type": "primary", "nodes": ["1", "2", "3"]}}')

        loader = FileLoader(str(csv), str(adj), str(meta))
        builder = GraphDatasetBuilder(loader, timestep=0)

        # Should have no edges since neither ends where the other starts
        assert builder.edge_index.size(1) == 0


class TestNodeFeatures:
    """Test node feature construction"""

    def test_x_is_tensor(self, graph_builder):
        """Test that node features x is a PyTorch tensor"""
        assert isinstance(graph_builder.x, torch.Tensor)

    def test_x_dtype(self, graph_builder):
        """Test that x has float dtype"""
        assert graph_builder.x.dtype == torch.float

    def test_x_shape(self, graph_builder):
        """Test that x has correct shape [num_nodes, num_features]"""
        num_nodes = len(graph_builder._edge_as_node_map)
        assert graph_builder.x.size(0) == num_nodes
        # 21 road types + 1 oneway = 22 features
        assert graph_builder.x.size(1) == 22

    def test_one_hot_encoding(self, graph_builder):
        """Test that road types are one-hot encoded"""
        # Each node should have exactly one 1 in the road type features (first 21)
        # or all zeros if road type not found
        for i in range(graph_builder.x.size(0)):
            road_type_features = graph_builder.x[i, :21]
            # Should be either all zeros or have exactly one 1
            ones_count = (road_type_features == 1).sum().item()
            assert ones_count in [0, 1]

    def test_oneway_feature(self, graph_builder):
        """Test that oneway feature is binary"""
        # Last feature is oneway
        oneway_features = graph_builder.x[:, -1]
        unique_values = torch.unique(oneway_features)
        # Should only contain 0s and/or 1s
        assert all(val in [0, 1] for val in unique_values.tolist())

    def test_feature_consistency(self, tmp_path):
        """Test that features match metadata"""
        csv = tmp_path / "travel.csv"
        csv.write_text("time_slot,edge0_traversal_time_sec\n00:05,10")

        adj = tmp_path / "adj.csv"
        adj.write_text("edge_id,vertex_start_id,vertex_end_id\n0,1,2")

        meta = tmp_path / "meta.json"
        meta.write_text('{"1": {"oneway": true, "road_type": "primary", "nodes": ["1", "2"]}}')

        loader = FileLoader(str(csv), str(adj), str(meta))
        builder = GraphDatasetBuilder(loader, timestep=0)

        # Edge 0 should have primary road type (index 2) and oneway=1
        features = builder.x[0]
        assert features[2] == 1  # primary is at index 2
        assert features[-1] == 1  # oneway is True

    def test_missing_metadata_defaults(self, tmp_path):
        """Test that edges without metadata get default features"""
        csv = tmp_path / "travel.csv"
        csv.write_text("time_slot,edge0_traversal_time_sec\n00:05,10")

        adj = tmp_path / "adj.csv"
        adj.write_text("edge_id,vertex_start_id,vertex_end_id\n0,1,2")

        meta = tmp_path / "meta.json"
        # Metadata doesn't include vertices 1 or 2
        meta.write_text('{"1": {"oneway": true, "road_type": "primary", "nodes": ["5", "6"]}}')

        loader = FileLoader(str(csv), str(adj), str(meta))
        builder = GraphDatasetBuilder(loader, timestep=0)

        # Should have all zeros for missing metadata
        features = builder.x[0]
        assert (features == 0).all()


class TestTargetTensor:
    """Test target tensor construction"""

    def test_y_is_tensor(self, graph_builder):
        """Test that target y is a PyTorch tensor"""
        assert isinstance(graph_builder.y, torch.Tensor)

    def test_y_dtype(self, graph_builder):
        """Test that y has float dtype"""
        assert graph_builder.y.dtype == torch.float

    def test_y_length(self, graph_builder):
        """Test that y has correct length (one value per node)"""
        num_nodes = len(graph_builder._edge_as_node_map)
        assert graph_builder.y.size(0) == num_nodes

    def test_y_values_match_timestep(self, file_loader):
        """Test that y values correspond to the specified timestep"""
        builder_t0 = GraphDatasetBuilder(file_loader, timestep=0)
        builder_t1 = GraphDatasetBuilder(file_loader, timestep=1)

        # Values should be different for different timesteps
        assert not torch.equal(builder_t0.y, builder_t1.y)

    def test_correct_travel_times_at_timestep_0(self, graph_builder):
        """Test that travel times match the CSV data at timestep 0 (00:00)"""
        # From Day 1 CSV at 00:00: edges 0-9 have values 120,85,95,110,75,130,90,105,80,125
        # Order depends on edge_as_node_map, but we know the values
        y_values = sorted(graph_builder.y.tolist())
        expected = sorted([120.0, 85.0, 95.0, 110.0, 75.0, 130.0, 90.0, 105.0, 80.0, 125.0])
        assert y_values == expected

    def test_correct_travel_times_at_timestep_1(self, file_loader):
        """Test that travel times match the CSV data at timestep 1 (01:00)"""
        builder = GraphDatasetBuilder(file_loader, timestep=1)
        # From Day 1 CSV at 01:00: edges 0-9 have values 115,80,90,105,70,125,85,100,75,120
        y_values = sorted(builder.y.tolist())
        expected = sorted([115.0, 80.0, 90.0, 105.0, 70.0, 125.0, 85.0, 100.0, 75.0, 120.0])
        assert y_values == expected

    def test_raises_on_invalid_timestep(self, file_loader):
        """Test that invalid timestep raises an error"""
        # 3 days × 12 timesteps = 36 total timesteps (0-35)
        with pytest.raises(IndexError):
            GraphDatasetBuilder(file_loader, timestep=100)


class TestGetData:
    """Test the get_data() method"""

    def test_returns_data_object(self, graph_builder):
        """Test that get_data returns a PyTorch Geometric Data object"""
        data = graph_builder.get_data()
        assert isinstance(data, Data)

    def test_data_has_required_attributes(self, graph_builder):
        """Test that Data object has x, edge_index, and y"""
        data = graph_builder.get_data()
        assert hasattr(data, 'x')
        assert hasattr(data, 'edge_index')
        assert hasattr(data, 'y')

    def test_data_attributes_match_builder(self, graph_builder):
        """Test that Data attributes match the builder's tensors"""
        data = graph_builder.get_data()

        assert torch.equal(data.x, graph_builder.x)
        assert torch.equal(data.edge_index, graph_builder.edge_index)
        assert torch.equal(data.y, graph_builder.y)

    def test_data_is_valid_pyg_object(self, graph_builder):
        """Test that Data object passes basic PyG validation"""
        data = graph_builder.get_data()

        # Check shapes are compatible
        num_nodes = data.x.size(0)
        assert data.y.size(0) == num_nodes
        if data.edge_index.size(1) > 0:
            # All edge indices should be valid node indices
            assert data.edge_index.max() < num_nodes
            assert data.edge_index.min() >= 0


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_disconnected_graph(self, tmp_path):
        """Test graph with no connections between edges"""
        csv = tmp_path / "travel.csv"
        csv.write_text("time_slot,edge0_traversal_time_sec,edge1_traversal_time_sec\n00:05,10,20")

        adj = tmp_path / "adj.csv"
        # Two disconnected edges
        adj.write_text("edge_id,vertex_start_id,vertex_end_id\n0,1,2\n1,10,20")

        meta = tmp_path / "meta.json"
        meta.write_text('{"1": {"oneway": true, "road_type": "primary", "nodes": ["1", "2", "10", "20"]}}')

        loader = FileLoader(str(csv), str(adj), str(meta))
        builder = GraphDatasetBuilder(loader, timestep=0)

        # Should still create nodes, but no edges between them
        assert len(builder._edge_as_node_map) == 2
        assert builder.edge_index.size(1) == 0

    def test_single_edge_graph(self, tmp_path):
        """Test graph with only one edge"""
        csv = tmp_path / "travel.csv"
        csv.write_text("time_slot,edge0_traversal_time_sec\n00:05,10")

        adj = tmp_path / "adj.csv"
        adj.write_text("edge_id,vertex_start_id,vertex_end_id\n0,1,2")

        meta = tmp_path / "meta.json"
        meta.write_text('{"1": {"oneway": true, "road_type": "primary", "nodes": ["1", "2"]}}')

        loader = FileLoader(str(csv), str(adj), str(meta))
        builder = GraphDatasetBuilder(loader, timestep=0)

        # Should create one node, no edges
        assert len(builder._edge_as_node_map) == 1
        assert builder.edge_index.size(1) == 0
        assert builder.x.size(0) == 1
        assert builder.y.size(0) == 1

    def test_filters_edges_without_travel_data(self, tmp_path):
        """Test that edges without travel data are filtered out"""
        csv = tmp_path / "travel.csv"
        # Only has edge0 and edge2
        csv.write_text("time_slot,edge0_traversal_time_sec,edge2_traversal_time_sec\n00:05,10,30")

        adj = tmp_path / "adj.csv"
        # Has edges 0, 1, and 2
        adj.write_text("edge_id,vertex_start_id,vertex_end_id\n0,1,2\n1,2,3\n2,3,4")

        meta = tmp_path / "meta.json"
        meta.write_text('{"1": {"oneway": true, "road_type": "primary", "nodes": ["1", "2", "3", "4"]}}')

        loader = FileLoader(str(csv), str(adj), str(meta))
        builder = GraphDatasetBuilder(loader, timestep=0)

        # Should only include edges 0 and 2
        assert len(builder._edge_as_node_map) == 2
        assert 0 in builder._edge_as_node_map
        assert 2 in builder._edge_as_node_map
        assert 1 not in builder._edge_as_node_map

#pragma once

#include <sq_matrix.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <thread>
#include <unordered_map>
#include <vector>

namespace sph
{

using scalar = float;
using Vec3 = dixelu::point<scalar, 3>;

constexpr scalar pi = 3.14159265358979323846f;
constexpr scalar numerical_epsilon = 5e-3f;

[[nodiscard]] scalar wendland_c2(scalar distance, scalar smoothing_length);
[[nodiscard]] scalar wendland_c2(const Vec3& displacement, scalar smoothing_length);
[[nodiscard]] Vec3 wendland_c2_gradient(const Vec3& displacement, scalar smoothing_length);

struct Particle
{
	Vec3 position{};
	Vec3 velocity{};
	Vec3 acceleration{};
	scalar mass = 1.f;
	scalar smoothing_length = 1.f;
	scalar energy = 1.f;
	scalar cfl_time = 1.f;
	std::uint32_t interaction_count = 1;
};

struct SimulationConfig
{
	scalar maximum_time_step = 0.004f;
	scalar gravitational_constant = 0.001f;
	scalar heat_capacity_ratio = 5.f / 3.f;
	scalar polytropic_exponent = 5.f / 3.f;
	scalar polytropic_strength = 0.1f;
	scalar barnes_hut_theta = 0.55f;
	scalar courant_number = 0.3f;
	scalar minimum_smoothing_fraction = 0.05f;
	int desired_neighbors = 48;
	bool enable_hydrodynamics = true;
	bool enable_gravity = true;
};

struct InitialConditions
{
	std::size_t particle_count = 50000;
	std::uint32_t seed = 1;
	scalar domain_size = 100.f;
	scalar cloud_radius_fraction = 0.30f;
	scalar total_mass = 400000.f;
	scalar angular_speed = 0.10f;
	scalar initial_energy = 1.f;
};

enum class Field
{
	density,
	energy,
	speed,
	acceleration,
	x_velocity,
	y_velocity,
	z_velocity
};

struct RenderParticle
{
	Vec3 position{};
	scalar density = 0.f;
	scalar energy = 0.f;
	scalar speed = 0.f;
	scalar acceleration = 0.f;
	Vec3 velocity{};
};

struct SimulationSnapshot
{
	std::vector<RenderParticle> particles;
	scalar total_time = 0.f;
	scalar last_time_step = 0.f;
	scalar step_milliseconds = 0.f;
	scalar preparation_milliseconds = 0.f;
	scalar octree_milliseconds = 0.f;
	scalar neighbor_graph_milliseconds = 0.f;
	scalar density_milliseconds = 0.f;
	scalar force_milliseconds = 0.f;
	std::uint64_t step = 0;
	std::size_t octree_nodes = 0;
	std::size_t occupied_cells = 0;
	scalar maximum_density = 0.f;
	scalar maximum_speed = 0.f;
	scalar maximum_acceleration = 0.f;
	scalar minimum_smoothing_length = 0.f;
	scalar maximum_smoothing_length = 0.f;
	scalar smoothing_length_floor = 0.f;
	scalar maximum_sound_speed = 0.f;
	scalar average_interactions = 0.f;
	Vec3 momentum{};
	bool finite = true;
};

[[nodiscard]] std::vector<Particle> make_rotating_cloud(const InitialConditions& settings);

class Simulation3D
{
public:
	explicit Simulation3D(
		std::vector<Particle> particles,
		scalar reference_domain_size,
		SimulationConfig config = {});

	void set_config(const SimulationConfig& config);
	[[nodiscard]] const SimulationConfig& config() const noexcept { return config_; }
	[[nodiscard]] const std::vector<Particle>& particles() const noexcept { return particles_; }
	[[nodiscard]] const std::vector<scalar>& densities() const noexcept { return densities_; }
	[[nodiscard]] scalar total_time() const noexcept { return total_time_; }
	[[nodiscard]] std::uint64_t step_count() const noexcept { return step_count_; }
	[[nodiscard]] std::size_t octree_node_count() const noexcept;
	[[nodiscard]] std::size_t occupied_cell_count() const noexcept;

	void prepare();
	[[nodiscard]] scalar step();
	[[nodiscard]] SimulationSnapshot make_snapshot(
		scalar step_milliseconds = 0.f,
		scalar preparation_milliseconds = 0.f) const;

	[[nodiscard]] static Vec3 pair_gravity(
		const Particle& accelerated,
		const Particle& source,
		scalar gravitational_constant);

private:
	struct CellKey
	{
		std::int64_t x = 0;
		std::int64_t y = 0;
		std::int64_t z = 0;
		bool operator==(const CellKey&) const = default;
	};

	struct CellKeyHash
	{
		[[nodiscard]] std::size_t operator()(const CellKey& key) const noexcept;
	};

	class NeighborGrid
	{
	public:
		void build(const std::vector<Particle>& particles, scalar fallback_cell_size);
		void collect(
			const Vec3& position,
			scalar radius,
			std::vector<std::uint32_t>& result) const;
		void clear() noexcept { cells_.clear(); }
		[[nodiscard]] std::size_t occupied_cell_count() const noexcept { return cells_.size(); }

	private:
		[[nodiscard]] CellKey key_for(const Vec3& position) const;
		scalar cell_size_ = 1.f;
		std::unordered_map<CellKey, std::vector<std::uint32_t>, CellKeyHash> cells_;
	};

	struct OctreeNode
	{
		Vec3 center{};
		scalar half_size = 0.f;
		scalar mass = 0.f;
		Vec3 center_of_mass{};
		scalar maximum_smoothing_length = 0.f;
		std::uint32_t begin = 0;
		std::uint32_t end = 0;
		std::array<std::int32_t, 8> children{-1, -1, -1, -1, -1, -1, -1, -1};
		bool leaf = true;
	};

	void build_octree();
	void build_neighbor_graph();
	void prepare_state(bool prepare_gravity, bool prepare_hydrodynamics);
	std::int32_t build_octree_node(
		std::uint32_t begin,
		std::uint32_t end,
		const Vec3& center,
		scalar half_size,
		int depth);
	[[nodiscard]] Vec3 gravity_for(
		std::uint32_t particle_index,
		std::vector<std::int32_t>& traversal) const;
	[[nodiscard]] scalar pressure_for(scalar density, scalar energy) const;
	[[nodiscard]] scalar minimum_smoothing_length() const noexcept;

	std::vector<Particle> particles_;
	std::vector<Particle> next_particles_;
	std::vector<scalar> densities_;
	std::vector<scalar> pressures_;
	std::vector<std::uint32_t> neighbor_offsets_;
	std::vector<std::uint32_t> neighbors_;
	std::vector<scalar> neighbor_weights_;
	std::vector<std::uint32_t> octree_indices_;
	std::vector<std::uint32_t> octree_scratch_;
	std::vector<OctreeNode> octree_;
	NeighborGrid neighbor_grid_;
	SimulationConfig config_;
	scalar reference_domain_size_ = 100.f;
	scalar reference_smoothing_length_ = 1.f;
	scalar total_time_ = 0.f;
	scalar last_time_step_ = 0.f;
	std::uint64_t step_count_ = 0;
	scalar last_preparation_milliseconds_ = 0.f;
	scalar last_octree_milliseconds_ = 0.f;
	scalar last_neighbor_graph_milliseconds_ = 0.f;
	scalar last_density_milliseconds_ = 0.f;
	scalar last_force_milliseconds_ = 0.f;
};

class SimulationRunner
{
public:
	explicit SimulationRunner(
		InitialConditions initial = {},
		SimulationConfig config = {});
	~SimulationRunner();

	SimulationRunner(const SimulationRunner&) = delete;
	SimulationRunner& operator=(const SimulationRunner&) = delete;

	void set_running(bool running) noexcept;
	[[nodiscard]] bool is_running() const noexcept;
	void request_single_step() noexcept;
	void request_reset(const InitialConditions& initial);
	void set_config(const SimulationConfig& config);
	[[nodiscard]] SimulationConfig config() const;
	[[nodiscard]] std::shared_ptr<const SimulationSnapshot> snapshot() const;

private:
	void worker_loop();
	void publish_snapshot(scalar step_ms = 0.f, scalar preparation_ms = 0.f);

	std::unique_ptr<Simulation3D> simulation_;
	mutable std::mutex command_mutex_;
	SimulationConfig requested_config_;
	std::optional<InitialConditions> pending_reset_;
	std::atomic<bool> running_{false};
	std::atomic<bool> single_step_{false};
	std::atomic<bool> stop_{false};
	std::atomic<std::shared_ptr<const SimulationSnapshot>> snapshot_;
	std::thread worker_;
};

} // namespace sph

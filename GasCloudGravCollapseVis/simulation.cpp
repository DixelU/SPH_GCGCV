#include "pch.h"
#include "simulation.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <numeric>
#include <utility>

namespace sph
{
namespace
{

[[nodiscard]] std::size_t parallel_worker_count(std::size_t count)
{
	constexpr std::size_t minimum_items_per_worker = 1024;
	const std::size_t available_workers = std::max(1u, std::thread::hardware_concurrency());
	const std::size_t useful_workers =
		std::max<std::size_t>(1, (count + minimum_items_per_worker - 1) / minimum_items_per_worker);
	return std::min(available_workers, useful_workers);
}

template <class Function>
void parallel_for_blocks_indexed(
	std::size_t count,
	std::size_t worker_count,
	Function&& function)
{
	if (count == 0)
		return;
	if (worker_count == 1)
	{
		function(std::size_t{0}, count, std::size_t{0});
		return;
	}

	std::vector<std::jthread> workers;
	workers.reserve(worker_count - 1);
	for (std::size_t worker = 1; worker < worker_count; ++worker)
	{
		workers.emplace_back([&, worker]
		{
			function(
				count * worker / worker_count,
				count * (worker + 1) / worker_count,
				worker);
		});
	}
	function(0, count / worker_count, 0);
}

template <class Function>
void parallel_for_blocks(std::size_t count, Function&& function)
{
	parallel_for_blocks_indexed(
		count,
		parallel_worker_count(count),
		[&](std::size_t begin, std::size_t end, std::size_t)
		{
			function(begin, end);
		});
}

[[nodiscard]] bool finite(const Vec3& value)
{
	return std::isfinite(value[0]) &&
		std::isfinite(value[1]) &&
		std::isfinite(value[2]);
}

[[nodiscard]] std::uint64_t mix64(std::uint64_t value) noexcept
{
	value ^= value >> 30;
	value *= UINT64_C(0xbf58476d1ce4e5b9);
	value ^= value >> 27;
	value *= UINT64_C(0x94d049bb133111eb);
	value ^= value >> 31;
	return value;
}

[[nodiscard]] scalar cube(scalar value)
{
	return value * value * value;
}

} // namespace

scalar wendland_c2(scalar distance, scalar smoothing_length)
{
	if (!(smoothing_length > 0.f) || distance < 0.f || distance >= smoothing_length)
		return 0.f;
	const scalar q = distance / smoothing_length;
	const scalar one_minus_q = 1.f - q;
	const scalar one_minus_q2 = one_minus_q * one_minus_q;
	return (21.f / (2.f * pi)) * one_minus_q2 * one_minus_q2 *
		(1.f + 4.f * q) / cube(smoothing_length);
}

scalar wendland_c2(const Vec3& displacement, scalar smoothing_length)
{
	return wendland_c2(displacement.get_norm(), smoothing_length);
}

Vec3 wendland_c2_gradient(const Vec3& displacement, scalar smoothing_length)
{
	if (!(smoothing_length > 0.f))
		return {};
	const scalar distance_squared = displacement.get_norm2();
	if (!(distance_squared > 0.f) || distance_squared >= smoothing_length * smoothing_length)
		return {};
	const scalar q = std::sqrt(distance_squared) / smoothing_length;
	const scalar one_minus_q = 1.f - q;
	const scalar h2 = smoothing_length * smoothing_length;
	const scalar h5 = h2 * h2 * smoothing_length;
	return (-210.f / pi) * one_minus_q * one_minus_q * one_minus_q *
		displacement / h5;
}

std::vector<Particle> make_rotating_cloud(const InitialConditions& settings)
{
	std::vector<Particle> particles;
	particles.reserve(settings.particle_count);
	if (settings.particle_count == 0)
		return particles;

	std::mt19937 generator(settings.seed);
	std::uniform_real_distribution<scalar> coordinate(-1.f, 1.f);
	const scalar domain_size = std::max(settings.domain_size, 1.f);
	const scalar cloud_radius = domain_size *
		std::clamp(settings.cloud_radius_fraction, 0.01f, 0.49f);
	const scalar volume = (4.f / 3.f) * pi * cube(cloud_radius);
	const scalar number_density = static_cast<scalar>(settings.particle_count) / volume;
	const scalar support_volume = 48.f / std::max(number_density, numerical_epsilon);
	const scalar smoothing_length = std::cbrt(
		support_volume / ((4.f / 3.f) * pi));
	const scalar particle_mass = std::max(settings.total_mass, numerical_epsilon) /
		static_cast<scalar>(settings.particle_count);

	while (particles.size() < settings.particle_count)
	{
		const Vec3 unit_candidate{
			coordinate(generator),
			coordinate(generator),
			coordinate(generator)};
		if (unit_candidate.get_norm2() > 1.f)
			continue;
		const Vec3 position = cloud_radius * unit_candidate;
		const Vec3 velocity{
			-settings.angular_speed * position[1],
			 settings.angular_speed * position[0],
			 0.f};
		particles.push_back(Particle{
			.position = position,
			.velocity = velocity,
			.acceleration = {},
			.mass = particle_mass,
			.smoothing_length = smoothing_length,
			.energy = std::max(settings.initial_energy, numerical_epsilon),
			.cfl_time = 10.f,
			.interaction_count = 1});
	}
	return particles;
}

std::size_t Simulation3D::CellKeyHash::operator()(const CellKey& key) const noexcept
{
	const std::uint64_t x = mix64(static_cast<std::uint64_t>(key.x));
	const std::uint64_t y = mix64(static_cast<std::uint64_t>(key.y) + UINT64_C(0x9e3779b97f4a7c15));
	const std::uint64_t z = mix64(static_cast<std::uint64_t>(key.z) + UINT64_C(0x3c79ac492ba7b653));
	return static_cast<std::size_t>(x ^ std::rotl(y, 21) ^ std::rotl(z, 42));
}

Simulation3D::CellKey Simulation3D::NeighborGrid::key_for(const Vec3& position) const
{
	return {
		static_cast<std::int64_t>(std::floor(position[0] / cell_size_)),
		static_cast<std::int64_t>(std::floor(position[1] / cell_size_)),
		static_cast<std::int64_t>(std::floor(position[2] / cell_size_))};
}

void Simulation3D::NeighborGrid::build(
	const std::vector<Particle>& particles,
	scalar fallback_cell_size)
{
	std::vector<scalar> smoothing_lengths;
	smoothing_lengths.reserve(particles.size());
	for (const Particle& particle : particles)
		if (std::isfinite(particle.smoothing_length) && particle.smoothing_length > 0.f)
			smoothing_lengths.push_back(particle.smoothing_length);
	cell_size_ = std::max(fallback_cell_size, numerical_epsilon);
	if (!smoothing_lengths.empty())
	{
		const auto median = smoothing_lengths.begin() + smoothing_lengths.size() / 2;
		std::nth_element(smoothing_lengths.begin(), median, smoothing_lengths.end());
		cell_size_ = std::max(cell_size_, *median);
	}
	cells_.clear();
	cells_.reserve(particles.size());
	for (std::uint32_t index = 0; index < particles.size(); ++index)
		cells_[key_for(particles[index].position)].push_back(index);
}

void Simulation3D::NeighborGrid::collect(
	const Vec3& position,
	scalar radius,
	std::vector<std::uint32_t>& result) const
{
	result.clear();
	const CellKey center = key_for(position);
	const std::int64_t reach = std::max<std::int64_t>(
		1,
		static_cast<std::int64_t>(std::ceil(std::max(radius, 0.f) / cell_size_)));
	for (std::int64_t z = -reach; z <= reach; ++z)
		for (std::int64_t y = -reach; y <= reach; ++y)
			for (std::int64_t x = -reach; x <= reach; ++x)
			{
				const auto found = cells_.find({center.x + x, center.y + y, center.z + z});
				if (found != cells_.end())
					result.insert(result.end(), found->second.begin(), found->second.end());
			}
}

void Simulation3D::build_neighbor_graph()
{
	neighbor_grid_.build(particles_, minimum_smoothing_length());
	const std::size_t worker_count = parallel_worker_count(particles_.size());
	using neighbor_pair = std::pair<std::uint32_t, std::uint32_t>;
	std::vector<std::vector<neighbor_pair>> pair_partitions(worker_count);
	parallel_for_blocks_indexed(
		particles_.size(),
		worker_count,
		[&](std::size_t block_begin, std::size_t block_end, std::size_t worker)
		{
			auto& pairs = pair_partitions[worker];
			pairs.clear();
			pairs.reserve(
				(block_end - block_begin) *
				static_cast<std::size_t>(config_.desired_neighbors) / 2);
			std::vector<std::uint32_t> candidates;
			candidates.reserve(static_cast<std::size_t>(config_.desired_neighbors) * 2);
			for (std::uint32_t source_index = static_cast<std::uint32_t>(block_begin);
				source_index < block_end;
				++source_index)
			{
				const Particle& source = particles_[source_index];
				neighbor_grid_.collect(
					source.position,
					source.smoothing_length,
					candidates);
				for (std::uint32_t candidate_index : candidates)
				{
					if (candidate_index == source_index)
						continue;
					const Particle& candidate = particles_[candidate_index];
					// The particle with the larger support owns the pair. Equal
					// supports use index order. This discovers max(h_i, h_j)
					// interactions exactly once without making a single outlier
					// inflate every spatial-hash bucket.
					if (source.smoothing_length < candidate.smoothing_length ||
						(source.smoothing_length == candidate.smoothing_length &&
							source_index > candidate_index))
						continue;
					const Vec3 displacement = source.position - candidate.position;
					if (displacement.get_norm2() <=
						source.smoothing_length * source.smoothing_length)
						pairs.emplace_back(source_index, candidate_index);
				}
			}
		});

	std::vector<std::uint32_t> degrees(particles_.size(), 1);
	for (const auto& partition : pair_partitions)
		for (const auto& [first, second] : partition)
		{
			++degrees[first];
			++degrees[second];
		}
	neighbor_offsets_.assign(particles_.size() + 1, 0);
	for (std::size_t index = 0; index < particles_.size(); ++index)
		neighbor_offsets_[index + 1] = neighbor_offsets_[index] + degrees[index];
	neighbors_.resize(neighbor_offsets_.back());
	neighbor_weights_.resize(neighbor_offsets_.back());
	std::vector<std::uint32_t> cursors(
		neighbor_offsets_.begin(),
		neighbor_offsets_.end() - 1);
	for (std::uint32_t index = 0; index < particles_.size(); ++index)
	{
		const std::uint32_t offset = cursors[index]++;
		neighbors_[offset] = index;
		neighbor_weights_[offset] = wendland_c2(0.f, particles_[index].smoothing_length);
	}
	for (const auto& partition : pair_partitions)
		for (const auto& [first, second] : partition)
		{
			const scalar distance = (
				particles_[first].position - particles_[second].position).get_norm();
			const std::uint32_t first_offset = cursors[first]++;
			const std::uint32_t second_offset = cursors[second]++;
			neighbors_[first_offset] = second;
			neighbors_[second_offset] = first;
			// Density is a directed estimate: rho_i uses h_i. The symmetric
			// graph is only a broad-phase superset needed by both endpoints.
			neighbor_weights_[first_offset] = wendland_c2(
				distance,
				particles_[first].smoothing_length);
			neighbor_weights_[second_offset] = wendland_c2(
				distance,
				particles_[second].smoothing_length);
		}
}

Simulation3D::Simulation3D(
	std::vector<Particle> particles,
	scalar reference_domain_size,
	SimulationConfig config) :
	particles_(std::move(particles)),
	config_(config),
	reference_domain_size_(std::max(reference_domain_size, 1.f))
{
	std::vector<scalar> initial_smoothing_lengths;
	initial_smoothing_lengths.reserve(particles_.size());
	for (const Particle& particle : particles_)
		if (std::isfinite(particle.smoothing_length) && particle.smoothing_length > 0.f)
			initial_smoothing_lengths.push_back(particle.smoothing_length);
	if (!initial_smoothing_lengths.empty())
	{
		const auto median = initial_smoothing_lengths.begin() +
			initial_smoothing_lengths.size() / 2;
		std::nth_element(
			initial_smoothing_lengths.begin(),
			median,
			initial_smoothing_lengths.end());
		reference_smoothing_length_ = *median;
	}
	set_config(config);
}

void Simulation3D::set_config(const SimulationConfig& config)
{
	config_ = config;
	config_.maximum_time_step = std::clamp(config_.maximum_time_step, 1e-8f, 1.f);
	config_.gravitational_constant = std::max(config_.gravitational_constant, 0.f);
	config_.heat_capacity_ratio = std::max(config_.heat_capacity_ratio, 1.0001f);
	config_.polytropic_exponent = std::max(config_.polytropic_exponent, 1.0001f);
	config_.polytropic_strength = std::max(config_.polytropic_strength, 0.f);
	config_.barnes_hut_theta = std::clamp(config_.barnes_hut_theta, 0.05f, 1.f);
	config_.courant_number = std::clamp(config_.courant_number, 0.01f, 0.9f);
	config_.minimum_smoothing_fraction = std::clamp(
		config_.minimum_smoothing_fraction,
		0.001f,
		0.5f);
	config_.desired_neighbors = std::clamp(config_.desired_neighbors, 8, 256);
}

std::size_t Simulation3D::octree_node_count() const noexcept
{
	return octree_.size();
}

std::size_t Simulation3D::occupied_cell_count() const noexcept
{
	return neighbor_grid_.occupied_cell_count();
}

scalar Simulation3D::minimum_smoothing_length() const noexcept
{
	const scalar absolute_floor = numerical_epsilon * reference_domain_size_ * 0.1f;
	const scalar resolution_floor =
		config_.minimum_smoothing_fraction * reference_smoothing_length_;
	return std::max(absolute_floor, resolution_floor);
}

void Simulation3D::build_octree()
{
	octree_.clear();
	octree_indices_.resize(particles_.size());
	octree_scratch_.resize(particles_.size());
	std::iota(octree_indices_.begin(), octree_indices_.end(), std::uint32_t{0});
	if (particles_.empty())
		return;

	Vec3 minimum = particles_.front().position;
	Vec3 maximum = minimum;
	for (const Particle& particle : particles_)
		for (std::size_t axis = 0; axis < 3; ++axis)
		{
			minimum[axis] = std::min(minimum[axis], particle.position[axis]);
			maximum[axis] = std::max(maximum[axis], particle.position[axis]);
		}
	const Vec3 center = (minimum + maximum) * 0.5f;
	scalar half_size = minimum_smoothing_length();
	for (std::size_t axis = 0; axis < 3; ++axis)
		half_size = std::max(half_size, (maximum[axis] - minimum[axis]) * 0.5001f);
	octree_.reserve(std::max<std::size_t>(8, particles_.size() / 2));
	build_octree_node(0, static_cast<std::uint32_t>(particles_.size()), center, half_size, 0);
}

std::int32_t Simulation3D::build_octree_node(
	std::uint32_t begin,
	std::uint32_t end,
	const Vec3& center,
	scalar half_size,
	int depth)
{
	const std::int32_t node_index = static_cast<std::int32_t>(octree_.size());
	octree_.push_back({});
	OctreeNode& inserted = octree_.back();
	inserted.center = center;
	inserted.half_size = half_size;
	inserted.begin = begin;
	inserted.end = end;

	scalar total_mass = 0.f;
	Vec3 weighted_position{};
	scalar maximum_smoothing = 0.f;
	for (std::uint32_t offset = begin; offset < end; ++offset)
	{
		const Particle& particle = particles_[octree_indices_[offset]];
		total_mass += particle.mass;
		weighted_position += particle.mass * particle.position;
		maximum_smoothing = std::max(maximum_smoothing, particle.smoothing_length);
	}
	inserted.mass = total_mass;
	inserted.center_of_mass = total_mass > 0.f ? weighted_position / total_mass : center;
	inserted.maximum_smoothing_length = maximum_smoothing;

	constexpr std::uint32_t leaf_capacity = 8;
	constexpr int maximum_depth = 24;
	if (end - begin <= leaf_capacity || depth >= maximum_depth || half_size <= minimum_smoothing_length() * 1e-3f)
		return node_index;

	std::array<std::uint32_t, 8> counts{};
	auto octant_for = [&](std::uint32_t particle_index)
	{
		const Vec3& position = particles_[particle_index].position;
		return static_cast<std::uint32_t>(
			(position[0] >= center[0] ? 1 : 0) |
			(position[1] >= center[1] ? 2 : 0) |
			(position[2] >= center[2] ? 4 : 0));
	};
	for (std::uint32_t offset = begin; offset < end; ++offset)
		counts[octant_for(octree_indices_[offset])]++;

	std::array<std::uint32_t, 8> starts{};
	starts[0] = begin;
	for (std::size_t octant = 1; octant < 8; ++octant)
		starts[octant] = starts[octant - 1] + counts[octant - 1];
	auto cursors = starts;
	for (std::uint32_t offset = begin; offset < end; ++offset)
	{
		const std::uint32_t particle_index = octree_indices_[offset];
		octree_scratch_[cursors[octant_for(particle_index)]++] = particle_index;
	}
	std::copy(
		octree_scratch_.begin() + begin,
		octree_scratch_.begin() + end,
		octree_indices_.begin() + begin);

	octree_[node_index].leaf = false;
	const scalar child_half_size = half_size * 0.5f;
	for (std::size_t octant = 0; octant < 8; ++octant)
	{
		if (counts[octant] == 0)
			continue;
		Vec3 child_center = center;
		child_center[0] += (octant & 1) ? child_half_size : -child_half_size;
		child_center[1] += (octant & 2) ? child_half_size : -child_half_size;
		child_center[2] += (octant & 4) ? child_half_size : -child_half_size;
		octree_[node_index].children[octant] = build_octree_node(
			starts[octant],
			starts[octant] + counts[octant],
			child_center,
			child_half_size,
			depth + 1);
	}
	return node_index;
}

Vec3 Simulation3D::pair_gravity(
	const Particle& accelerated,
	const Particle& source,
	scalar gravitational_constant)
{
	const Vec3 displacement = source.position - accelerated.position;
	const scalar softening = std::max(
		0.5f * (source.smoothing_length + accelerated.smoothing_length),
		numerical_epsilon);
	const scalar softened_distance_squared =
		displacement.get_norm2() + softening * softening;
	return gravitational_constant * source.mass * displacement /
		(softened_distance_squared * std::sqrt(softened_distance_squared));
}

Vec3 Simulation3D::gravity_for(
	std::uint32_t particle_index,
	std::vector<std::int32_t>& traversal) const
{
	Vec3 acceleration{};
	if (!config_.enable_gravity || octree_.empty())
		return acceleration;
	const Particle& target = particles_[particle_index];
	traversal.clear();
	traversal.push_back(0);
	while (!traversal.empty())
	{
		const OctreeNode& node = octree_[traversal.back()];
		traversal.pop_back();
		if (!(node.mass > 0.f))
			continue;
		if (node.leaf)
		{
			for (std::uint32_t offset = node.begin; offset < node.end; ++offset)
			{
				const std::uint32_t source_index = octree_indices_[offset];
				if (source_index != particle_index)
					acceleration += pair_gravity(
						target,
						particles_[source_index],
						config_.gravitational_constant);
			}
			continue;
		}

		const Vec3 displacement = node.center_of_mass - target.position;
		const scalar distance_squared = displacement.get_norm2();
		bool contains_target = true;
		for (std::size_t axis = 0; axis < 3; ++axis)
			contains_target = contains_target &&
				std::abs(target.position[axis] - node.center[axis]) <= node.half_size;
		const scalar width = node.half_size * 2.f;
		if (!contains_target && distance_squared > 0.f &&
			width * width < config_.barnes_hut_theta * config_.barnes_hut_theta * distance_squared)
		{
			const scalar softening = std::max(
				0.5f * (node.maximum_smoothing_length + target.smoothing_length),
				numerical_epsilon);
			const scalar softened_distance_squared =
				distance_squared + softening * softening;
			acceleration += config_.gravitational_constant * node.mass * displacement /
				(softened_distance_squared * std::sqrt(softened_distance_squared));
		}
		else
		{
			for (std::int32_t child : node.children)
				if (child >= 0)
					traversal.push_back(child);
		}
	}
	return acceleration;
}

scalar Simulation3D::pressure_for(scalar density, scalar energy) const
{
	const scalar safe_density = std::max(density, numerical_epsilon);
	const scalar ideal = (config_.heat_capacity_ratio - 1.f) *
		safe_density * std::max(energy, numerical_epsilon);
	const scalar polytropic = config_.polytropic_strength *
		std::pow(safe_density, config_.polytropic_exponent);
	return std::max(ideal + polytropic, numerical_epsilon);
}

void Simulation3D::prepare()
{
	prepare_state(config_.enable_gravity, config_.enable_hydrodynamics);
}

void Simulation3D::prepare_state(
	bool prepare_gravity,
	bool prepare_hydrodynamics)
{
	const auto preparation_begin = std::chrono::steady_clock::now();
	const auto octree_begin = preparation_begin;
	if (prepare_gravity)
		build_octree();
	else
	{
		octree_.clear();
		octree_indices_.clear();
	}
	const auto octree_end = std::chrono::steady_clock::now();
	last_octree_milliseconds_ = static_cast<scalar>(
		std::chrono::duration<double, std::milli>(octree_end - octree_begin).count());

	if (prepare_hydrodynamics)
	{
		const auto graph_begin = std::chrono::steady_clock::now();
		build_neighbor_graph();
		const auto graph_end = std::chrono::steady_clock::now();
		last_neighbor_graph_milliseconds_ = static_cast<scalar>(
			std::chrono::duration<double, std::milli>(graph_end - graph_begin).count());

		densities_.assign(particles_.size(), 0.f);
		pressures_.assign(particles_.size(), 0.f);
		parallel_for_blocks(particles_.size(), [&](std::size_t block_begin, std::size_t block_end)
		{
			for (std::size_t index = block_begin; index < block_end; ++index)
			{
				const Particle& source = particles_[index];
				scalar density = 0.f;
				for (std::uint32_t offset = neighbor_offsets_[index];
					offset < neighbor_offsets_[index + 1];
					++offset)
				{
					const Particle& candidate = particles_[neighbors_[offset]];
					density += candidate.mass * neighbor_weights_[offset];
				}
				densities_[index] = std::max(density, numerical_epsilon);
				pressures_[index] = pressure_for(densities_[index], source.energy);
			}
		});
		last_density_milliseconds_ = static_cast<scalar>(
			std::chrono::duration<double, std::milli>(
				std::chrono::steady_clock::now() - graph_end).count());
	}
	else
	{
		neighbor_offsets_.clear();
		neighbors_.clear();
		neighbor_weights_.clear();
		neighbor_grid_.clear();
		densities_.assign(particles_.size(), 0.f);
		pressures_.assign(particles_.size(), 0.f);
		last_neighbor_graph_milliseconds_ = 0.f;
		last_density_milliseconds_ = 0.f;
	}

	last_preparation_milliseconds_ = static_cast<scalar>(
		std::chrono::duration<double, std::milli>(
			std::chrono::steady_clock::now() - preparation_begin).count());
}

scalar Simulation3D::step()
{
	prepare_state(config_.enable_gravity, config_.enable_hydrodynamics);
	if (particles_.empty())
	{
		last_force_milliseconds_ = 0.f;
		return 0.f;
	}

	scalar time_step = config_.maximum_time_step;
	for (const Particle& particle : particles_)
		if (std::isfinite(particle.cfl_time) && particle.cfl_time > 0.f)
			time_step = std::min(time_step, particle.cfl_time);
	time_step = std::max(time_step, 1e-8f);
	next_particles_.resize(particles_.size());
	const auto force_begin = std::chrono::steady_clock::now();

	parallel_for_blocks(particles_.size(), [&](std::size_t block_begin, std::size_t block_end)
	{
		std::vector<std::int32_t> gravity_traversal;
		gravity_traversal.reserve(256);
		for (std::size_t index = block_begin; index < block_end; ++index)
		{
			const Particle& current = particles_[index];
			const scalar density_i = config_.enable_hydrodynamics ? densities_[index] : 1.f;
			const scalar pressure_i = config_.enable_hydrodynamics ? pressures_[index] : 0.f;
			const scalar sound_speed_i = config_.enable_hydrodynamics ? std::sqrt(
				config_.polytropic_exponent * pressure_i / density_i) : 0.f;
			Vec3 pressure_acceleration{};
			scalar energy_rate = 0.f;
			scalar maximum_mu = 0.f;
			std::uint32_t interactions = config_.enable_hydrodynamics ?
				1u : current.interaction_count;

			if (config_.enable_hydrodynamics)
			{
				for (std::uint32_t offset = neighbor_offsets_[index];
					offset < neighbor_offsets_[index + 1];
					++offset)
				{
					const std::uint32_t candidate_index = neighbors_[offset];
					if (candidate_index == index)
						continue;
					// Adapt h_i from neighbors inside i's own support. Counting the
					// entire symmetric broad phase lets large-h neighbors force h_i
					// downward even when they are outside its kernel.
					if (neighbor_weights_[offset] > 0.f)
						++interactions;
					const Particle& neighbor = particles_[candidate_index];
					const Vec3 position_difference = current.position - neighbor.position;
					const scalar smoothing_length =
						0.5f * (current.smoothing_length + neighbor.smoothing_length);
					const scalar distance_squared = position_difference.get_norm2();
					if (distance_squared > smoothing_length * smoothing_length)
						continue;
					const Vec3 velocity_difference = current.velocity - neighbor.velocity;
					const scalar density_j = densities_[candidate_index];
					const scalar pressure_j = pressures_[candidate_index];
					const scalar sound_speed_j = std::sqrt(
						config_.polytropic_exponent * pressure_j / density_j);
					const scalar velocity_position_product =
						velocity_difference * position_difference;
					scalar viscosity = 0.f;
					if (velocity_position_product < 0.f)
					{
						scalar mu = smoothing_length * velocity_position_product /
							(distance_squared + 0.02f * smoothing_length * smoothing_length);
						const scalar average_sound_speed =
							0.5f * (sound_speed_i + sound_speed_j);
						mu = std::max(mu, -2.f * average_sound_speed);
						const scalar average_density = 0.5f * (density_i + density_j);
						viscosity = (-0.5f * average_sound_speed * mu + mu * mu) /
							std::max(average_density, numerical_epsilon);
						viscosity = std::min(viscosity, 1.f);
						maximum_mu = std::max(maximum_mu, -mu);
					}

					const Vec3 gradient = wendland_c2_gradient(
						position_difference,
						smoothing_length);
					const scalar common_pressure =
						pressure_i / (density_i * density_i) +
						pressure_j / (density_j * density_j);
					pressure_acceleration += neighbor.mass *
						(common_pressure + viscosity) * gradient;
					energy_rate += neighbor.mass *
						(velocity_difference * gradient) *
						(common_pressure + 0.5f * viscosity);
				}
			}

			Particle updated = current;
			updated.acceleration = gravity_for(
				static_cast<std::uint32_t>(index),
				gravity_traversal) - pressure_acceleration;
			updated.velocity += time_step * updated.acceleration;
			updated.position += time_step * updated.velocity;
			if (config_.enable_hydrodynamics)
				updated.energy = std::max(
					current.energy + time_step * energy_rate,
					numerical_epsilon);
			updated.interaction_count = interactions;

			if (config_.enable_hydrodynamics)
			{
				const scalar target_ratio = static_cast<scalar>(config_.desired_neighbors) /
					static_cast<scalar>(std::max<std::uint32_t>(interactions, 1));
				const scalar target_smoothing = current.smoothing_length * std::cbrt(target_ratio);
				updated.smoothing_length = std::clamp(
					0.5f * (current.smoothing_length + target_smoothing),
					minimum_smoothing_length(),
					reference_domain_size_ * 0.5f);
			}

			const scalar resolved_length = std::max(
				updated.smoothing_length,
				minimum_smoothing_length());
			const scalar gravity_time = config_.courant_number * std::sqrt(
				resolved_length /
				std::max(updated.acceleration.get_norm(), numerical_epsilon));
			const scalar signal_time = config_.courant_number * resolved_length /
				std::max(
					updated.velocity.get_norm() + sound_speed_i +
						1.2f * (sound_speed_i + maximum_mu),
					numerical_epsilon);
			updated.cfl_time = std::max(
				std::min(gravity_time, signal_time),
				1e-8f);

			if (!finite(updated.position) || !finite(updated.velocity) ||
				!finite(updated.acceleration) || !std::isfinite(updated.energy) ||
				!std::isfinite(updated.smoothing_length) || !std::isfinite(updated.cfl_time))
			{
				updated = current;
				updated.velocity = {};
				updated.acceleration = {};
				updated.cfl_time = config_.maximum_time_step;
			}
			next_particles_[index] = updated;
		}
	});
	last_force_milliseconds_ = static_cast<scalar>(
		std::chrono::duration<double, std::milli>(
			std::chrono::steady_clock::now() - force_begin).count());

	particles_.swap(next_particles_);
	total_time_ += time_step;
	last_time_step_ = time_step;
	++step_count_;
	return time_step;
}

SimulationSnapshot Simulation3D::make_snapshot(
	scalar step_milliseconds,
	scalar preparation_milliseconds) const
{
	SimulationSnapshot result;
	result.particles.reserve(particles_.size());
	result.total_time = total_time_;
	result.last_time_step = last_time_step_;
	result.step_milliseconds = step_milliseconds;
	result.preparation_milliseconds = preparation_milliseconds > 0.f ?
		preparation_milliseconds : last_preparation_milliseconds_;
	result.octree_milliseconds = last_octree_milliseconds_;
	result.neighbor_graph_milliseconds = last_neighbor_graph_milliseconds_;
	result.density_milliseconds = last_density_milliseconds_;
	result.force_milliseconds = last_force_milliseconds_;
	result.step = step_count_;
	result.octree_nodes = octree_.size();
	result.occupied_cells = neighbor_grid_.occupied_cell_count();
	result.smoothing_length_floor = minimum_smoothing_length();
	scalar total_mass = 0.f;
	double total_interactions = 0.;
	for (std::size_t index = 0; index < particles_.size(); ++index)
	{
		const Particle& particle = particles_[index];
		const scalar density = index < densities_.size() ? densities_[index] : 0.f;
		const scalar speed = particle.velocity.get_norm();
		const scalar acceleration = particle.acceleration.get_norm();
		result.particles.push_back({
			.position = particle.position,
			.density = density,
			.energy = particle.energy,
			.speed = speed,
			.acceleration = acceleration,
			.velocity = particle.velocity});
		result.maximum_density = std::max(result.maximum_density, density);
		result.maximum_speed = std::max(result.maximum_speed, speed);
		result.maximum_acceleration = std::max(result.maximum_acceleration, acceleration);
		if (index == 0)
			result.minimum_smoothing_length = particle.smoothing_length;
		else
			result.minimum_smoothing_length = std::min(
				result.minimum_smoothing_length,
				particle.smoothing_length);
		result.maximum_smoothing_length = std::max(
			result.maximum_smoothing_length,
			particle.smoothing_length);
		if (index < densities_.size() && index < pressures_.size() &&
			densities_[index] > 0.f)
			result.maximum_sound_speed = std::max(
				result.maximum_sound_speed,
				std::sqrt(config_.polytropic_exponent * pressures_[index] /
					densities_[index]));
		total_interactions += particle.interaction_count;
		result.momentum += particle.mass * particle.velocity;
		total_mass += particle.mass;
		result.finite = result.finite && finite(particle.position) &&
			finite(particle.velocity) && finite(particle.acceleration) &&
			std::isfinite(particle.mass) && std::isfinite(particle.smoothing_length) &&
			std::isfinite(particle.energy);
	}
	if (!particles_.empty())
		result.average_interactions = static_cast<scalar>(
			total_interactions / static_cast<double>(particles_.size()));
	(void)total_mass;
	return result;
}

SimulationRunner::SimulationRunner(
	InitialConditions initial,
	SimulationConfig config) :
	requested_config_(config)
{
	simulation_ = std::make_unique<Simulation3D>(
		make_rotating_cloud(initial),
		initial.domain_size,
		config);
	simulation_->prepare();
	publish_snapshot();
	worker_ = std::thread(&SimulationRunner::worker_loop, this);
}

SimulationRunner::~SimulationRunner()
{
	stop_.store(true, std::memory_order_release);
	if (worker_.joinable())
		worker_.join();
}

void SimulationRunner::set_running(bool running) noexcept
{
	running_.store(running, std::memory_order_release);
}

bool SimulationRunner::is_running() const noexcept
{
	return running_.load(std::memory_order_acquire);
}

void SimulationRunner::request_single_step() noexcept
{
	single_step_.store(true, std::memory_order_release);
}

void SimulationRunner::request_reset(const InitialConditions& initial)
{
	std::lock_guard lock(command_mutex_);
	pending_reset_ = initial;
}

void SimulationRunner::set_config(const SimulationConfig& config)
{
	std::lock_guard lock(command_mutex_);
	requested_config_ = config;
}

SimulationConfig SimulationRunner::config() const
{
	std::lock_guard lock(command_mutex_);
	return requested_config_;
}

std::shared_ptr<const SimulationSnapshot> SimulationRunner::snapshot() const
{
	return snapshot_.load(std::memory_order_acquire);
}

void SimulationRunner::publish_snapshot(scalar step_ms, scalar preparation_ms)
{
	auto next = std::make_shared<SimulationSnapshot>(
		simulation_->make_snapshot(step_ms, preparation_ms));
	snapshot_.store(std::move(next), std::memory_order_release);
}

void SimulationRunner::worker_loop()
{
	while (!stop_.load(std::memory_order_acquire))
	{
		std::optional<InitialConditions> reset;
		SimulationConfig config;
		{
			std::lock_guard lock(command_mutex_);
			reset.swap(pending_reset_);
			config = requested_config_;
		}
		if (reset)
		{
			simulation_ = std::make_unique<Simulation3D>(
				make_rotating_cloud(*reset),
				reset->domain_size,
				config);
			simulation_->prepare();
			publish_snapshot();
		}
		simulation_->set_config(config);

		const bool advance = running_.load(std::memory_order_acquire) ||
			single_step_.exchange(false, std::memory_order_acq_rel);
		if (!advance)
		{
			std::this_thread::sleep_for(std::chrono::milliseconds(8));
			continue;
		}

		const auto begin = std::chrono::steady_clock::now();
		(void)simulation_->step();
		const scalar step_ms = static_cast<scalar>(
			std::chrono::duration<double, std::milli>(
				std::chrono::steady_clock::now() - begin).count());
		publish_snapshot(step_ms);
	}
}

} // namespace sph

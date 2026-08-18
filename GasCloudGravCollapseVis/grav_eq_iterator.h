#pragma once

#include "consts.h"
#include "buffered_queue_spsc.h"
#include "field_vis.h"
#include <sq_matrix.h>
#include "weird_hacks.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#include <queue>
#include <stack>

#include "allocator.h"

#define is_variable_timestep // uncomment to push it working again...
//#define measuring_performance

using current_float_t = float;

using namespace std;
using point = dixelu::point<current_float_t, 2>;

namespace _____type_desc
{
auto p_zero = [&](current_float_t t, const point& x) -> point { return point(); };
auto d_zero = [&](current_float_t t, current_float_t x) -> current_float_t { return 0.; };
}

using p_lambda = std::function<point(current_float_t, const point&)>;
using d_lambda = std::function<current_float_t(current_float_t, current_float_t)>;

#define _x(p) ((p)[0])
#define _y(p) ((p)[1])
#define _z(p) ((p)[2])

namespace grav_eq_utils
{
inline bool point_in_square(const point& lb_sq, const point& rt_sq, const point& p_pos)
{
	point center = (rt_sq + lb_sq) * 0.5f;
	current_float_t width = (_x(rt_sq) - _x(lb_sq)) * 0.5f;
	return (abs(_x(center) - _x(p_pos)) < width) && (abs(_y(center) - _y(p_pos)) < width);
}
constexpr current_float_t epsilon = 5e-3;
constexpr current_float_t pi = 3.14159265358979323846f;

// Two-dimensional Wendland C2 kernel with compact support r < h.
// Its normalization scales as h^-2, and its vanishing central gradient avoids
// the pairing instability encouraged by kernels with a finite gradient at r=0.
inline current_float_t pressure_core(const point& r, current_float_t h)
{
	if (h <= 0)
		return 0.;
	const current_float_t q = r.get_norm() / h;
	if (q >= 1.f)
		return 0.;
	const current_float_t one_minus_q = 1.f - q;
	const current_float_t one_minus_q2 = one_minus_q * one_minus_q;
	return (7.f / pi) * one_minus_q2 * one_minus_q2 *
		(1.f + 4.f * q) / (h * h);
}
inline point pressure_core_gradient(const point& r, current_float_t h)
{
	if (h <= 0)
		return {0,0};
	const current_float_t r2 = r.get_norm2();
	if (r2 <= 0.f || r2 >= h * h)
		return {0,0};
	const current_float_t q = std::sqrt(r2) / h;
	const current_float_t one_minus_q = 1.f - q;
	return (-140.f / pi) * one_minus_q * one_minus_q *
		one_minus_q * r / (h * h * h * h);
}
inline current_float_t pressure_core(current_float_t r, current_float_t h)
{
	if (h <= 0 || r < 0 || r >= h)
		return 0.;
	const current_float_t q = r / h;
	const current_float_t one_minus_q = 1.f - q;
	const current_float_t one_minus_q2 = one_minus_q * one_minus_q;
	return (7.f / pi) * one_minus_q2 * one_minus_q2 *
		(1.f + 4.f * q) / (h * h);
}
inline current_float_t inverse_pressure_core(current_float_t d, current_float_t h)
{
	if (h <= 0 || d >= pressure_core(0.f, h))
		return 0.f;
	if (d <= 0)
		return h;

	current_float_t lower = 0.f;
	current_float_t upper = h;
	for (int iteration = 0; iteration < 24; iteration++)
	{
		const current_float_t middle = (lower + upper) * 0.5f;
		if (pressure_core(middle, h) > d)
			lower = middle;
		else
			upper = middle;
	}
	return (lower + upper) * 0.5f;
}
};

inline void draw_smooth_circle(const float x, const float y, const float r, const float value, const float dvalue, const float dangle)
{
	float begin = grav_eq_utils::pressure_core(0, r);
	for (float val = begin; val > 0.005f; val /= dvalue)
	{

		float rad = grav_eq_utils::inverse_pressure_core(val, r);

		auto [pr, pg, pb] = get_color(val * value);

		glColor4f(pr, pg, pb, 0.025f);

		glBegin(GL_POLYGON);

		for (float i = 0.f; i < 360.f; i += dangle)
			glVertex2f(rad * cos(ANGTORAD(i)) + x, rad * sin(ANGTORAD(i)) + y);

		glEnd();

	}
}

struct particle
{
	// Includes the particle itself. A larger two-dimensional stencil reduces
	// visible lattice locking and gives the pressure estimate useful support.
	static constexpr int desired_amount_of_interactions = 24;
	int interactions_count;
	point position;
	point velocity;
	point acceleration;
	current_float_t mass;
	current_float_t radius;
	current_float_t energy;
#ifdef is_variable_timestep
	current_float_t cfl_time;
#endif
	bool visited;
	particle(point position = {0.f, 0.f}, point velocity = {0.f, 0.f}, point acceleration = {0.f, 0.f}, current_float_t part_mass = 0., current_float_t radius = 0., current_float_t energy = 0., int amount_of_interactions = 1
#ifdef is_variable_timestep
		, current_float_t cfl_time = 10
#endif
	) :
		position(position), velocity(velocity), mass(part_mass), radius(radius), energy(energy), acceleration(acceleration), interactions_count(amount_of_interactions)
#ifdef is_variable_timestep
		, cfl_time(cfl_time)
#endif
	{
		visited = false;
	}
	inline bool operator==(const particle& prt) const
	{
		using namespace grav_eq_utils;
		return ((position - prt.position).get_norm2() == 0) && ((velocity - prt.velocity).get_norm2() == 0);
	}
	//inverse to operator-
	inline particle operator+(const particle& prt) const
	{
		current_float_t ratio = mass / (prt.mass + mass);
		current_float_t aratio = 1. - ratio;
		return particle(
			ratio * position + aratio * prt.position,
			ratio * velocity + aratio * prt.velocity,
			ratio * acceleration + aratio * prt.acceleration,
			prt.mass + mass,
			sqrt(radius * radius + prt.radius * prt.radius),
			energy + prt.energy,
			interactions_count + prt.interactions_count
#ifdef is_variable_timestep
			, (std::min)(cfl_time, prt.cfl_time)
#endif
		);
	}
	//inverse to operator+
	/*
	inline particle operator-(const particle& prt) const {
		current_float_t ratio = mass / (mass - prt.mass);
		return particle(
			ratio * (position - prt.position) + prt.position,
			ratio * (velocity - prt.velocity) + prt.velocity,
			ratio * (acceleration - prt.acceleration) + prt.acceleration,
			mass - prt.mass,
			sqrt(radius * radius - prt.radius * prt.radius),
			energy - prt.energy,
			interactions_count - prt.interactions_count
#ifdef is_variable_timestep
			, cfl_time
#endif
		);
	}*/
	inline particle operator+=(const particle& prt)
	{
		return ((*this) = (*this) + prt);
	}
	/*inline particle operator-=(const particle& prt) {
		return ((*this) = (*this) - prt);
	}*/
};


namespace draw_type
{
enum class dt
{
	density, energy, x_speed, y_speed, x_acceleration, y_acceleration
};
}


struct node
{
	enum positioning
	{
		leftbottom = 0, lefttop = 1, righttop = 2, rightbottom = 3, null = 4
	};
	int particles_count_in_subtrees;
	uint32_t spatial_index;
	particle mass_center;
	node* left_bottom;
	node* left_top;
	node* right_bottom;
	node* right_top;
	node* parent;//for backward tracing.
	node* null_node;
	point leftbottom_corner;
	point righttop_corner;
	node()
	{
		null_node = nullptr;
		left_bottom = left_top = right_bottom = right_top = parent = nullptr;
		particles_count_in_subtrees = 0;
		spatial_index = (std::numeric_limits<uint32_t>::max)();
		leftbottom_corner = righttop_corner = {0, 0};
		mass_center = particle();
	}
	node(node* parent, positioning pos_id) :node()
	{
		this->parent = parent;
		point center = (parent->righttop_corner + parent->leftbottom_corner) / 2.;
		mass_center.position = center;
		point local_shift{0.f, _y(center - parent->leftbottom_corner)};
		switch (pos_id)
		{
			case leftbottom:
				parent->left_bottom = this;
				leftbottom_corner = parent->leftbottom_corner;
				righttop_corner = center;
				break;
			case lefttop:
				parent->left_top = this;
				righttop_corner = center + local_shift;
				leftbottom_corner = parent->leftbottom_corner + local_shift;
				break;
			case righttop:
				parent->right_top = this;
				righttop_corner = parent->righttop_corner;
				leftbottom_corner = center;
				break;
			case rightbottom:
				parent->right_bottom = this;
				leftbottom_corner = center - local_shift;
				righttop_corner = parent->righttop_corner - local_shift;
				break;
			case null:
				throw std::exception("Constructed beyond meaningful area!", (int)parent);
				break;
		}
	}
	node(node* parent, const particle& p) : node(parent, get_positioning(parent, p.position))
	{
		mass_center = p;
	}
	inline static positioning get_positioning(node* nd, point pos)
	{
		if (!nd)
			return null;
		if (!nd->point_is_inside(pos))
		{
			pos[0] = std::clamp(pos[0], nd->leftbottom_corner[0], nd->righttop_corner[0]);
			pos[1] = std::clamp(pos[1], nd->leftbottom_corner[1], nd->righttop_corner[1]);
		}
		point center = (nd->righttop_corner + nd->leftbottom_corner) / 2.;
		point difference = pos - center;
		if (_x(difference) >= 0)
		{
			if (_y(difference) >= 0)
				return righttop;
			else
				return rightbottom;
		}
		else
		{
			if (_y(difference) >= 0)
				return lefttop;
			else
				return leftbottom;
		}
	}
	inline node** get_dptr(positioning D)
	{
		switch (D)
		{
			case leftbottom:
				return &this->left_bottom;
			case lefttop:
				return &this->left_top;
			case righttop:
				return &this->right_top;
			case rightbottom:
				return &this->right_bottom;
			default:
				//throw std::exception("null node was called");
				return &this->null_node;
		}
	}
	inline bool point_is_inside(const point& pos)
	{
		return (leftbottom_corner <= pos && pos <= righttop_corner);
	}
	inline void zero_pointers()
	{
		null_node = nullptr;
		left_bottom = left_top = right_bottom = right_top = parent = nullptr;
	}
	inline node*& get(positioning D)
	{
		return *get_dptr(D);
	}
};

using vecnode = std::vector<node*>;

struct grav_eq_iteration_buffers
{
	vecnode subtree_traversal;
	vecnode radial_nodes;
	vecnode gravity_traversal;
};

struct sph_neighbor_grid
{
	struct sampled_field_grid
	{
		uint32_t cells_per_side = 1;
		uint32_t x_begin = 0;
		uint32_t y_begin = 0;
		uint32_t width = 0;
		uint32_t height = 0;
		current_float_t cell_size = 0.f;
		std::vector<current_float_t> values;

		inline current_float_t at(uint32_t x, uint32_t y) const
		{
			return values[static_cast<size_t>(y) * width + x];
		}
	};

	struct grid_entry
	{
		uint64_t key;
		uint32_t particle_index;
	};

	struct cell_range
	{
		uint64_t key;
		uint32_t begin;
		uint32_t end;
	};

	static constexpr uint32_t invalid_index =
		(std::numeric_limits<uint32_t>::max)();
	static constexpr int coordinate_bits = 29;
	static constexpr uint64_t coordinate_mask =
		(uint64_t{1} << coordinate_bits) - 1;

	vecnode particle_nodes;
	vecnode tree_traversal;
	std::vector<uint8_t> particle_levels;
	std::vector<grid_entry> entries;
	std::vector<grid_entry> entry_scratch;
	std::vector<cell_range> cells;
	std::vector<uint32_t> hash_slots;
	std::vector<std::vector<std::pair<uint32_t, uint32_t>>> pair_partitions;
	std::vector<uint32_t> neighbor_offsets;
	std::vector<uint32_t> neighbors;
	std::vector<current_float_t> neighbor_weights;
	std::vector<current_float_t> cached_densities;
	std::vector<current_float_t> cached_energies;
	std::vector<uint32_t> degrees;
	std::vector<uint32_t> cursors;
	point domain_leftbottom = {0.f, 0.f};
	current_float_t domain_size = 0.f;
	current_float_t finest_cell_size = 0.f;
	uint8_t finest_level_bits = 0;
	uint8_t largest_occupied_level = 0;

	inline static uint64_t mix_key(uint64_t value)
	{
		value ^= value >> 33;
		value *= UINT64_C(0xff51afd7ed558ccd);
		value ^= value >> 33;
		value *= UINT64_C(0xc4ceb9fe1a85ec53);
		value ^= value >> 33;
		return value;
	}

	inline static size_t preparation_worker_count(size_t particle_count)
	{
		constexpr size_t minimum_particles_per_worker = 2048;
		const size_t useful_workers =
			(particle_count + minimum_particles_per_worker - 1) /
			minimum_particles_per_worker;
		return (std::max)(size_t{1}, (std::min)(
			useful_workers,
			static_cast<size_t>((std::max)(
				std::thread::hardware_concurrency(),
				1u))));
	}

	template <class function_t>
	inline static void parallel_for_blocks(
		size_t item_count,
		size_t worker_count,
		function_t&& function)
	{
		if (worker_count <= 1)
		{
			function(size_t{0}, item_count, size_t{0});
			return;
		}

		std::vector<std::jthread> workers;
		workers.reserve(worker_count - 1);
		for (size_t worker = 1; worker < worker_count; worker++)
		{
			workers.emplace_back([&, worker]()
			{
				function(
					item_count * worker / worker_count,
					item_count * (worker + 1) / worker_count,
					worker);
			});
		}
		function(size_t{0}, item_count / worker_count, size_t{0});
	}

	inline static uint64_t make_key(
		uint8_t level,
		uint32_t x,
		uint32_t y)
	{
		return (uint64_t{level} << (coordinate_bits * 2)) |
			((uint64_t{x} & coordinate_mask) << coordinate_bits) |
			(uint64_t{y} & coordinate_mask);
	}

	inline uint32_t cell_count_at(uint8_t level) const
	{
		return uint32_t{1} << (finest_level_bits - level);
	}

	inline current_float_t cell_size_at(uint8_t level) const
	{
		return std::ldexp(finest_cell_size, level);
	}

	inline uint32_t coordinate_at(
		current_float_t value,
		current_float_t minimum,
		current_float_t cell_size,
		uint32_t cell_count) const
	{
		const auto raw_coordinate = static_cast<int64_t>(
			std::floor((value - minimum) / cell_size));
		return static_cast<uint32_t>((std::clamp)(
			raw_coordinate,
			int64_t{0},
			static_cast<int64_t>(cell_count) - 1));
	}

	inline const cell_range* find_cell(uint64_t key) const
	{
		if (hash_slots.empty())
			return nullptr;
		const size_t mask = hash_slots.size() - 1;
		size_t slot = static_cast<size_t>(mix_key(key)) & mask;
		while (hash_slots[slot] != invalid_index)
		{
			const cell_range& cell = cells[hash_slots[slot]];
			if (cell.key == key)
				return &cell;
			slot = (slot + 1) & mask;
		}
		return nullptr;
	}

	inline void collect_particle_nodes(node* root)
	{
		particle_nodes.clear();
		tree_traversal.clear();
		if (!root ||
			(!root->particles_count_in_subtrees &&
				std::abs(root->mass_center.mass) <= grav_eq_utils::epsilon))
			return;

		tree_traversal.push_back(root);
		while (!tree_traversal.empty())
		{
			node* current_node = tree_traversal.back();
			tree_traversal.pop_back();
			if (!current_node->particles_count_in_subtrees)
			{
				current_node->spatial_index =
					static_cast<uint32_t>(particle_nodes.size());
				particle_nodes.push_back(current_node);
				continue;
			}

			for (int position = node::positioning::leftbottom;
				position < node::positioning::null;
				position++)
			{
				if (node* child = current_node->get(
					static_cast<node::positioning>(position)))
					tree_traversal.push_back(child);
			}
		}
	}

	inline void build_cell_table()
	{
		// Build and compact the hash table in linear time. Entries retain their
		// particle-index order within each cell, as they did after the old sort.
		cells.clear();
		size_t slot_count = 8;
		while (slot_count < entries.size() * 2)
			slot_count *= 2;
		hash_slots.assign(slot_count, invalid_index);
		const size_t mask = slot_count - 1;
		degrees.resize(entries.size());
		cells.reserve(entries.size());
		for (uint32_t entry_index = 0;
			entry_index < entries.size();
			entry_index++)
		{
			size_t slot =
				static_cast<size_t>(mix_key(entries[entry_index].key)) & mask;
			while (hash_slots[slot] != invalid_index &&
				cells[hash_slots[slot]].key != entries[entry_index].key)
				slot = (slot + 1) & mask;
			if (hash_slots[slot] == invalid_index)
			{
				hash_slots[slot] = static_cast<uint32_t>(cells.size());
				cells.push_back({entries[entry_index].key, 0, 0});
			}
			const uint32_t cell_index = hash_slots[slot];
			degrees[entry_index] = cell_index;
			cells[cell_index].end++;
		}

		uint32_t next_begin = 0;
		for (cell_range& cell : cells)
		{
			const uint32_t count = cell.end;
			cell.begin = next_begin;
			cell.end = next_begin + count;
			next_begin = cell.end;
		}
		cursors.resize(cells.size());
		for (uint32_t cell_index = 0;
			cell_index < cells.size();
			cell_index++)
			cursors[cell_index] = cells[cell_index].begin;
		entry_scratch.resize(entries.size());
		for (uint32_t entry_index = 0;
			entry_index < entries.size();
			entry_index++)
			entry_scratch[cursors[degrees[entry_index]]++] = entries[entry_index];
		entries.swap(entry_scratch);
	}

	inline void build_neighbor_graph(node* root)
	{
		collect_particle_nodes(root);
		const size_t particle_count = particle_nodes.size();
		particle_levels.resize(particle_count);
		entries.resize(particle_count);
		for (auto& partition : pair_partitions)
			partition.clear();
		neighbor_offsets.assign(particle_count + 1, 0);
		neighbors.clear();
		neighbor_weights.clear();
		cached_densities.clear();
		cached_energies.clear();
		if (!particle_count)
			return;

		domain_leftbottom = root->leftbottom_corner;
		domain_size =
			root->righttop_corner[0] - root->leftbottom_corner[0];
		const current_float_t minimum_resolved_diameter =
			(std::max)(domain_size * current_float_t{1e-6f},
				2.f * (*std::min_element(
					particle_nodes.begin(),
					particle_nodes.end(),
					[](const node* lhs, const node* rhs)
					{
						return lhs->mass_center.radius <
							rhs->mass_center.radius;
					}))->mass_center.radius);
		const current_float_t cell_ratio =
			(std::max)(domain_size / minimum_resolved_diameter, 1.f);
		finest_level_bits = static_cast<uint8_t>((std::clamp)(
			static_cast<int>(std::ceil(std::log2(cell_ratio))),
			0,
			24));
		finest_cell_size = std::ldexp(domain_size, -finest_level_bits);
		largest_occupied_level = 0;

		for (uint32_t particle_index = 0;
			particle_index < particle_count;
			particle_index++)
		{
			const particle& current_particle =
				particle_nodes[particle_index]->mass_center;
			const current_float_t diameter =
				(std::max)(2.f * current_particle.radius,
					finest_cell_size);
			const current_float_t level_ratio =
				(std::max)(diameter / finest_cell_size, 1.f);
			const uint8_t level = static_cast<uint8_t>((std::clamp)(
				static_cast<int>(std::ceil(std::log2(level_ratio))),
				0,
				static_cast<int>(finest_level_bits)));
			particle_levels[particle_index] = level;
			largest_occupied_level =
				(std::max)(largest_occupied_level, level);

			const current_float_t cell_size = cell_size_at(level);
			const uint32_t cell_count = cell_count_at(level);
			const uint32_t x = coordinate_at(
				current_particle.position[0],
				domain_leftbottom[0],
				cell_size,
				cell_count);
			const uint32_t y = coordinate_at(
				current_particle.position[1],
				domain_leftbottom[1],
				cell_size,
				cell_count);
			entries[particle_index] = {
				make_key(level, x, y),
				particle_index};
		}
		build_cell_table();

		const size_t worker_count = preparation_worker_count(particle_count);
		pair_partitions.resize(worker_count);
		parallel_for_blocks(
			particle_count,
			worker_count,
			[&](size_t begin, size_t end, size_t worker)
			{
				auto& pairs = pair_partitions[worker];
				pairs.clear();
				const size_t expected_pairs =
					(end - begin) * particle::desired_amount_of_interactions;
				if (pairs.capacity() < expected_pairs)
					pairs.reserve(expected_pairs);
				for (uint32_t first_index = static_cast<uint32_t>(begin);
					first_index < end;
					first_index++)
				{
					const particle& first =
						particle_nodes[first_index]->mass_center;
					const uint8_t first_level = particle_levels[first_index];
					for (uint8_t level = first_level;
						level <= largest_occupied_level;
						level++)
					{
						const current_float_t cell_size = cell_size_at(level);
						const uint32_t cell_count = cell_count_at(level);
						const uint32_t center_x = coordinate_at(
							first.position[0],
							domain_leftbottom[0],
							cell_size,
							cell_count);
						const uint32_t center_y = coordinate_at(
							first.position[1],
							domain_leftbottom[1],
							cell_size,
							cell_count);

						for (int y_offset = -1; y_offset <= 1; y_offset++)
							for (int x_offset = -1; x_offset <= 1; x_offset++)
							{
								const int64_t x =
									static_cast<int64_t>(center_x) + x_offset;
								const int64_t y =
									static_cast<int64_t>(center_y) + y_offset;
								if (x < 0 || y < 0 ||
									x >= cell_count || y >= cell_count)
									continue;
								const cell_range* cell = find_cell(make_key(
									level,
									static_cast<uint32_t>(x),
									static_cast<uint32_t>(y)));
								if (!cell)
									continue;

								for (uint32_t entry_index = cell->begin;
									entry_index < cell->end;
									entry_index++)
								{
									const uint32_t second_index =
										entries[entry_index].particle_index;
									if (level == first_level &&
										second_index <= first_index)
										continue;
									const particle& second =
										particle_nodes[second_index]->mass_center;
									const current_float_t support_radius =
										(std::max)(first.radius, second.radius);
									if ((first.position - second.position).get_norm2() <=
										support_radius * support_radius)
										pairs.push_back({first_index, second_index});
								}
							}
					}
				}
			});

		degrees.assign(particle_count, 1);
		for (const auto& partition : pair_partitions)
			for (const auto& [first, second] : partition)
			{
				degrees[first]++;
				degrees[second]++;
			}
		for (size_t index = 0; index < particle_count; index++)
			neighbor_offsets[index + 1] =
				neighbor_offsets[index] + degrees[index];
		neighbors.resize(neighbor_offsets.back());
		neighbor_weights.resize(neighbor_offsets.back());
		cursors.assign(neighbor_offsets.begin(), neighbor_offsets.end() - 1);
		for (uint32_t index = 0; index < particle_count; index++)
		{
			const uint32_t offset = cursors[index]++;
			neighbors[offset] = index;
			const particle& source = particle_nodes[index]->mass_center;
			neighbor_weights[offset] = grav_eq_utils::pressure_core(
				point{0.f, 0.f},
				source.radius);
		}
		for (const auto& partition : pair_partitions)
			for (const auto& [first, second] : partition)
			{
				const particle& first_particle =
					particle_nodes[first]->mass_center;
				const particle& second_particle =
					particle_nodes[second]->mass_center;
				const current_float_t support_radius = (std::max)(
					first_particle.radius,
					second_particle.radius);
				const current_float_t weight = grav_eq_utils::pressure_core(
					first_particle.position - second_particle.position,
					support_radius);
				// The kernel is symmetric, so calculate it once for both directions
				// and reuse it for the density and energy passes below.
				const uint32_t first_offset = cursors[first]++;
				const uint32_t second_offset = cursors[second]++;
				neighbors[first_offset] = second;
				neighbors[second_offset] = first;
				neighbor_weights[first_offset] = weight;
				neighbor_weights[second_offset] = weight;
			}

		cached_densities.assign(particle_count, 0.f);
		parallel_for_blocks(
			particle_count,
			worker_count,
			[&](size_t begin, size_t end, size_t)
			{
				for (uint32_t source_index = static_cast<uint32_t>(begin);
					source_index < end;
					source_index++)
				{
					current_float_t density = 0.f;
					for (uint32_t offset = neighbor_offsets[source_index];
						offset < neighbor_offsets[source_index + 1];
						offset++)
					{
						const particle& neighbor =
							particle_nodes[neighbors[offset]]->mass_center;
						density += neighbor.mass * neighbor_weights[offset];
					}
					cached_densities[source_index] = density;
				}
			});

		cached_energies.assign(particle_count, 0.f);
		parallel_for_blocks(
			particle_count,
			worker_count,
			[&](size_t begin, size_t end, size_t)
			{
				for (uint32_t source_index = static_cast<uint32_t>(begin);
					source_index < end;
					source_index++)
				{
					current_float_t energy = 0.f;
					for (uint32_t offset = neighbor_offsets[source_index];
						offset < neighbor_offsets[source_index + 1];
						offset++)
					{
						const uint32_t neighbor_index = neighbors[offset];
						const particle& neighbor =
							particle_nodes[neighbor_index]->mass_center;
						energy +=
							(neighbor.mass / cached_densities[neighbor_index]) *
							neighbor.energy * neighbor_weights[offset];
					}
					cached_energies[source_index] = energy;
				}
			});
	}

	inline void collect_neighbors(node* source, vecnode& results) const
	{
		results.clear();
		if (!source || source->spatial_index == invalid_index ||
			source->spatial_index + 1 >= neighbor_offsets.size())
			return;
		const uint32_t source_index = source->spatial_index;
		const uint32_t begin = neighbor_offsets[source_index];
		const uint32_t end = neighbor_offsets[source_index + 1];
		results.reserve(end - begin);
		for (uint32_t offset = begin; offset < end; offset++)
			results.push_back(particle_nodes[neighbors[offset]]);
	}

	inline current_float_t density_at(const node* source) const
	{
		return cached_densities[source->spatial_index];
	}

	inline current_float_t energy_at(const node* source) const
	{
		return cached_energies[source->spatial_index];
	}

	inline sampled_field_grid sample_field_grid(
		draw_type::dt type,
		int requested_depth,
		uint32_t requested_x_begin = 0,
		uint32_t requested_x_end = invalid_index,
		uint32_t requested_y_begin = 0,
		uint32_t requested_y_end = invalid_index) const
	{
		// Keep the grid dyadic so a sample cell exactly matches a quadtree cell
		// at the same depth. 24 also keeps the cell count representable by the
		// coordinate encoding used by the neighbour grid.
		const int depth = (std::clamp)(requested_depth, 0, 24);
		const uint32_t cells_per_side = uint32_t{1} << depth;
		const uint32_t x_begin = (std::min)(
			requested_x_begin,
			cells_per_side);
		const uint32_t y_begin = (std::min)(
			requested_y_begin,
			cells_per_side);
		const uint32_t x_end = (std::min)(
			requested_x_end,
			cells_per_side);
		const uint32_t y_end = (std::min)(
			requested_y_end,
			cells_per_side);

		sampled_field_grid result;
		result.cells_per_side = cells_per_side;
		result.x_begin = x_begin;
		result.y_begin = y_begin;
		result.width = x_end > x_begin ? x_end - x_begin : 0;
		result.height = y_end > y_begin ? y_end - y_begin : 0;
		result.cell_size = domain_size / cells_per_side;
		result.values.assign(
			static_cast<size_t>(result.width) * result.height,
			0.f);
		if (!result.width || !result.height ||
			particle_nodes.empty() || domain_size <= 0.f)
			return result;

		auto first_sample_at_or_after = [&](current_float_t coordinate)
		{
			return static_cast<int64_t>(std::ceil(
				(coordinate - domain_leftbottom[0]) / result.cell_size -
				0.5f));
		};
		auto last_sample_at_or_before = [&](current_float_t coordinate)
		{
			return static_cast<int64_t>(std::floor(
				(coordinate - domain_leftbottom[0]) / result.cell_size -
				0.5f));
		};

		for (uint32_t particle_index = 0;
			particle_index < particle_nodes.size();
			particle_index++)
		{
			const particle& source =
				particle_nodes[particle_index]->mass_center;
			if (!(source.radius > 0.f) || !std::isfinite(source.radius))
				continue;

			current_float_t sampled_property = 1.f;
			if (type != draw_type::dt::density)
			{
				if (particle_index >= cached_densities.size() ||
					!(cached_densities[particle_index] > 0.f))
					continue;
				switch (type)
				{
					case draw_type::dt::energy:
						sampled_property = source.energy;
						break;
					case draw_type::dt::x_speed:
						sampled_property = source.velocity[0];
						break;
					case draw_type::dt::y_speed:
						sampled_property = source.velocity[1];
						break;
					case draw_type::dt::x_acceleration:
						sampled_property = source.acceleration[0];
						break;
					case draw_type::dt::y_acceleration:
						sampled_property = source.acceleration[1];
						break;
					default:
						break;
				}
			}

			const current_float_t coefficient =
				type == draw_type::dt::density ?
				source.mass :
				(source.mass / cached_densities[particle_index]) *
					sampled_property;
			if (coefficient == 0.f || !std::isfinite(coefficient))
				continue;

			const int64_t particle_x_begin = (std::max)(
				first_sample_at_or_after(
					source.position[0] - source.radius),
				static_cast<int64_t>(x_begin));
			const int64_t particle_x_end = (std::min)(
				last_sample_at_or_before(
					source.position[0] + source.radius) + 1,
				static_cast<int64_t>(x_end));

			// The domain is square, so the same origin and spacing apply to Y.
			const int64_t particle_y_begin = (std::max)(
				static_cast<int64_t>(std::ceil(
					(source.position[1] - source.radius -
						domain_leftbottom[1]) / result.cell_size - 0.5f)),
				static_cast<int64_t>(y_begin));
			const int64_t particle_y_end = (std::min)(
				static_cast<int64_t>(std::floor(
					(source.position[1] + source.radius -
						domain_leftbottom[1]) / result.cell_size - 0.5f)) + 1,
				static_cast<int64_t>(y_end));
			if (particle_x_begin >= particle_x_end ||
				particle_y_begin >= particle_y_end)
				continue;

			for (int64_t y = particle_y_begin; y < particle_y_end; y++)
			{
				const current_float_t sample_y = domain_leftbottom[1] +
					(static_cast<current_float_t>(y) + 0.5f) *
					result.cell_size;
				for (int64_t x = particle_x_begin; x < particle_x_end; x++)
				{
					const current_float_t sample_x = domain_leftbottom[0] +
						(static_cast<current_float_t>(x) + 0.5f) *
						result.cell_size;
					const current_float_t weight =
						grav_eq_utils::pressure_core(
							point{
								sample_x - source.position[0],
								sample_y - source.position[1]},
							source.radius);
					if (weight == 0.f)
						continue;
					result.values[
						static_cast<size_t>(y - y_begin) * result.width +
						static_cast<size_t>(x - x_begin)] +=
						coefficient * weight;
				}
			}
		}

		return result;
	}
};

struct quad_tree
{
	//node*, node* = (temp,cur_node)
	using positioning = node::positioning;
	using node_buffer = buffered_queue_spsc<node, 1024>;

	node* root_node;
	std::unique_ptr<node_buffer> node_storage;
	recursive_mutex locker;
	recursive_mutex swap_prevention;
	quad_tree() :
		root_node(nullptr),
		node_storage(std::make_unique<node_buffer>())
	{
	}
	quad_tree(current_float_t size) : quad_tree()
	{
		root_node = &node_storage->emplace();
		root_node->leftbottom_corner = {-size * 0.5f, -size * 0.5f};
		root_node->righttop_corner = {size * 0.5f, size * 0.5f};
	}
	~quad_tree() = default;

	inline void clear()
	{
		if (!root_node)
			return;
		std::lock_guard<std::recursive_mutex> guard(locker);
		const point leftbottom_corner = root_node->leftbottom_corner;
		const point righttop_corner = root_node->righttop_corner;
		node_storage->clear();
		root_node = &node_storage->emplace();
		root_node->leftbottom_corner = leftbottom_corner;
		root_node->righttop_corner = righttop_corner;
	}

	inline void swap(quad_tree& tree)
	{
		swap_prevention.lock();
		tree.swap_prevention.lock();
		locker.lock();
		tree.locker.lock();

		std::swap(root_node, tree.root_node);
		std::swap(node_storage, tree.node_storage);

		tree.locker.unlock();
		locker.unlock();
		tree.swap_prevention.unlock();
		swap_prevention.unlock();
	}

	inline node** push(const particle& prt, unsigned char level = 0)
	{
		constexpr unsigned short max_level = 100;
		using namespace grav_eq_utils;
		node::positioning prt_pos = node::positioning::null;
		node** temp = nullptr;
		locker.lock();
		node* nd = root_node;
prp_begining:
		if (!nd || !nd->point_is_inside(prt.position))
			goto prp_ending;

		if (std::abs(nd->mass_center.mass) == 0 ||
			(nd->mass_center.position - prt.position).get_norm2() == 0 ||
			level >= max_level)
		{
			nd->mass_center += prt;
			goto prp_ending;
		}

		if (!nd->particles_count_in_subtrees)
		{
			node::positioning mc_pos = node::get_positioning(nd, nd->mass_center.position);
			node** temp = nd->get_dptr(mc_pos);
			if (!*temp)
			{
				*temp = &node_storage->emplace(nd, mc_pos);
				(*temp)->mass_center = nd->mass_center;
				//nd->particles_count_in_subtrees++;
			}
		}

		nd->mass_center += prt;
		nd->particles_count_in_subtrees++;
		prt_pos = node::get_positioning(nd, prt.position);
		temp = nd->get_dptr(prt_pos);
		if (!*temp)
			*temp = &node_storage->emplace(nd, prt_pos);
		nd = *temp;
		level++;
		goto prp_begining;
prp_ending:
		locker.unlock();
		return temp;
	}

	inline void draw(int draw_level, const point& center, current_float_t side_size, float points_size, float value_decrimemnt,
		draw_type::dt type = draw_type::dt::density, bool extra_flare = false, bool edge_drawer = false, bool draw_points = false, bool extended_draw = false)
	{
		swap_prevention.lock();
		std::vector<pair<node*, int>> cur_nodes;
		cur_nodes.reserve(1000);
		pair<node*, int> cur_node = {root_node , 0};
		node* temp = nullptr;
		current_float_t size = (_x(root_node->righttop_corner) - _x(root_node->leftbottom_corner));
		auto relative_size = side_size / size;
		auto scale = BEG_RANGE / RANGE;
		while (true)
		{
			if (cur_node.first)
			{
				if (cur_node.second < draw_level && cur_node.first->particles_count_in_subtrees
					//&& cur_node.first->righttop_corner >= lb_sc && cur_node.first->leftbottom_corner <= rt_sc // positioning on screen
					)
				{
					for (positioning i = positioning::leftbottom; i < positioning::null; ((int&)i)++)
					{
						if ((temp = cur_node.first->get(i)))
						{
							cur_nodes.push_back({temp, cur_node.second + 1});
						}
					}
				}
				else
				{
					point lb = (cur_node.first->leftbottom_corner * relative_size + center);
					point rt = (cur_node.first->righttop_corner * relative_size + center);
					current_float_t particle_value = 0;
					current_float_t node_value = 0;
					current_float_t ratio = (cur_node.first->mass_center.radius * cur_node.first->mass_center.radius) / std::pow(_x(cur_node.first->leftbottom_corner - cur_node.first->righttop_corner), 2);
					bool visited = cur_node.first->mass_center.visited;
					auto position = (cur_node.first->mass_center.position * relative_size + center);
					switch (type)
					{
						case draw_type::dt::density: // std::pow(_x(cur_node.first->leftbottom_corner - cur_node.first->righttop_corner), 2)
							particle_value = cur_node.first->mass_center.mass / (cur_node.first->mass_center.radius);
							node_value = particle_value * ratio;
							break;
						case draw_type::dt::energy:
							particle_value = cur_node.first->mass_center.energy;
							node_value = particle_value * ratio;
							break;
						case draw_type::dt::x_speed:
							particle_value = cur_node.first->mass_center.velocity[0];
							node_value = particle_value * ratio;
							break;
						case draw_type::dt::y_speed:
							particle_value = cur_node.first->mass_center.velocity[1];
							node_value = particle_value * ratio;
							break;
						case draw_type::dt::x_acceleration:
							particle_value = cur_node.first->mass_center.acceleration[0];
							node_value = particle_value * ratio;
							break;
						case draw_type::dt::y_acceleration:
							particle_value = cur_node.first->mass_center.acceleration[1];
							node_value = particle_value * ratio;
							break;
					}
					auto [nr, ng, nb] = get_color(node_value * value_decrimemnt);

					if (edge_drawer)
					{
						glBegin(GL_LINE_LOOP);
						glColor4f(nr, ng, nb, 1);
						glVertex2f(_x(lb), _y(lb));
						glVertex2f(_x(lb), _y(rt));
						glVertex2f(_x(rt), _y(rt));
						glVertex2f(_x(rt), _y(lb));
						glEnd();
					}
					// The extended mode is drawn as a sampled SPH grid by the
					// adapter. Keep this traversal only for optional quadtree-edge
					// and particle-point overlays.
					if (!extended_draw)
					{
						auto [pr, pg, pb] = get_color(particle_value * value_decrimemnt);
						auto a = (pr + pg + pb) * 0.15f;
						auto pointSize = (std::max)((current_float_t)1.f, cur_node.first->mass_center.radius * relative_size * scale);
						glPointSize(pointSize);
						glColor4f(pr, pg, pb, 0.05f + 0.05f * visited + a);
						glBegin(GL_POINTS);
						glVertex2f(_x(position), _y(position));
						glEnd();
					}

					if (draw_points)
					{
						glPointSize(points_size + extra_flare * points_size);
						glColor4f(0.5f + extra_flare * visited, 0.5f - extra_flare * visited, 0.5f + 0.5f * visited, 0.5f);
						glBegin(GL_POINTS);
						glVertex2f(_x(position), _y(position));
						glEnd();
					}
				}
			}
			if (cur_nodes.size())
			{
				cur_node = cur_nodes.back();
				cur_nodes.pop_back();
			}
			else
				break;
		}
		swap_prevention.unlock();
	}
};

class pooled_thread
{
public:
	enum class state
	{
		running, idle, waiting
	};
private:
	using funcT = std::function<void(void**)>;
	void* thread_data;//memory leak is allowed actually
	int await_in_milliseconds;
	funcT exec_func;
	std::atomic<bool> is_active;
	std::atomic<state> cur_state;
	std::atomic<state> default_state;
	std::recursive_mutex execution_locker;
	std::thread worker;
	void start_thread()
	{
		worker = std::thread([this]()
		{
			while (is_active.load(std::memory_order_acquire))
			{
				int sleep_time = 0;
				execution_locker.lock();
				if (cur_state.load(std::memory_order_relaxed) == state::waiting)
				{
					cur_state.store(state::running, std::memory_order_release);
					exec_func(&thread_data);
				}
				cur_state.store(default_state.load(std::memory_order_relaxed), std::memory_order_release);
				sleep_time = await_in_milliseconds;
				execution_locker.unlock();
				std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time));
			}
		});
	}
public:
	pooled_thread(funcT function = [](void** ptr) { return; }, int awaiting_time = 5) :
		exec_func(function), await_in_milliseconds(awaiting_time), is_active(true),
		cur_state(state::idle), default_state(state::idle)
	{
		thread_data = nullptr;
		start_thread();
	}
	~pooled_thread()
	{
		disable();
	}
	state get_state() const
	{
		return cur_state.load(std::memory_order_acquire);
	}
	void sign_awaiting()
	{
		execution_locker.lock();
		cur_state.store(state::waiting, std::memory_order_release);
		execution_locker.unlock();
	}
	void set_new_awaiting_time(int milliseconds)
	{
		execution_locker.lock();
		await_in_milliseconds = milliseconds;
		execution_locker.unlock();
	}
	void set_new_default_state(state def_state = state::idle)
	{
		execution_locker.lock();
		default_state.store(def_state, std::memory_order_release);
		execution_locker.unlock();
	}
	void set_new_function(funcT func)
	{
		execution_locker.lock();
		exec_func = func;
		execution_locker.unlock();
	}
	void disable()
	{
		is_active.store(false, std::memory_order_release);
		if (worker.joinable())
			worker.join();
	}
	void** __void_ptr_accsess()
	{
		return &thread_data;
	}
};

struct grav_eq_processor
{
	struct worker_thread_info
	{
		grav_eq_iteration_buffers buffers;
		int id = 0;
	};

	mutable vecnode _subdivision_roots;
	mutable std::atomic<size_t> _next_subdivision_root{0};
	mutable std::vector<pooled_thread*> threads;
	const size_t num_of_threads;
	decltype(std::chrono::steady_clock::now()) last_measure;

	current_float_t heat_capacity;
	current_float_t polytropic_coef;
	current_float_t time_step;
	current_float_t local_time_step;
	current_float_t total_time;
	const current_float_t __size;
	quad_tree current, buffer;
	sph_neighbor_grid spatial_neighbors;
	std::mutex buffer_mutex;
	std::mutex pre_swap;
	std::mutex pause;
	bool is_paused;
	bool flickering;
	bool reporting;
	bool halt_velocity;

	grav_eq_processor(const vector<particle>& input, current_float_t size) :
		current(size),
		buffer(size),
		heat_capacity(1.67f),
		polytropic_coef(1.67f),
		time_step(0.004f),
		flickering(false), reporting(false), halt_velocity(false),
		num_of_threads((std::max)(std::thread::hardware_concurrency() - 0, 1u)),
		__size(size),
		local_time_step(time_step),
		total_time(0)
#ifdef measuring_performance
		, last_iteration(std::chrono::high_resolution_clock::now())
#endif
	{
		is_paused = false;

		for (auto& prt : input)
			current.push(prt);
	}

	inline current_float_t get_density_at(node* source_node) const
	{
		return spatial_neighbors.density_at(source_node);
	}

	inline current_float_t get_energy_at(node* source_node) const
	{
		return spatial_neighbors.energy_at(source_node);
	}

	inline static point grav_force(const particle& center, const particle& distant_prt)
	{
		constexpr current_float_t grav_const = 0.001f;//just because ...
		const point displacement = distant_prt.position - center.position;
		const current_float_t softening_length = (std::max)(
			(distant_prt.radius + center.radius) * 0.5f,
			grav_eq_utils::epsilon);
		const current_float_t softened_distance_squared =
			displacement.get_norm2() + softening_length * softening_length;
		return grav_const * distant_prt.mass * displacement /
			std::pow(softened_distance_squared, 1.5f);
	}

	inline static point barnes_hutt_force_in_subtree(
		node* cur_node,
		const particle& current,
		const current_float_t error_edge_squared,
		vecnode* traversal_nodes)
	{
		point gravitational_force = {0, 0};
		traversal_nodes->clear();
		traversal_nodes->reserve(cur_node->particles_count_in_subtrees);
		constexpr bool is_real_gravity = false;
		auto get_squared_error = [](const particle& cur, node* check_node)
		{
			return 0.5f * (check_node->leftbottom_corner - check_node->righttop_corner).get_norm2() / (cur.position - check_node->mass_center.position).get_norm2();
		};

		node** ptemp;
		while (true)
		{
			if (cur_node)
			{
				if (cur_node->particles_count_in_subtrees &&
					(is_real_gravity || get_squared_error(current, cur_node) >= error_edge_squared))
				{
					for (node::positioning i = node::positioning::leftbottom; i < node::positioning::null; ((int&)i)++)
					{
						if (*(ptemp = cur_node->get_dptr(i)))
						{
							traversal_nodes->push_back(*ptemp);
						}
					}
				}
				else if ((current.position - cur_node->mass_center.position).get_norm2() >= pow(grav_eq_utils::epsilon, 2))
				{
					gravitational_force +=
						grav_force(current, cur_node->mass_center);
				}
			}
			if (!traversal_nodes->empty())
			{
				cur_node = traversal_nodes->back();
				traversal_nodes->pop_back();
			}
			else
				break;
		}
		return gravitational_force;
	}

	inline static current_float_t get_pressure__old(current_float_t density, current_float_t energy, current_float_t polytropic_coef, current_float_t heat_capacity)
	{
		constexpr current_float_t big_C_coef = 8.3f;
		return (heat_capacity - 1) * density * energy + big_C_coef * (polytropic_coef / 3.f + 1.f - heat_capacity) * std::pow(std::abs(density), polytropic_coef / 3.f + 1.f);
	}

	inline static current_float_t get_pressure(current_float_t density, current_float_t energy, current_float_t polytropic_coef, current_float_t heat_capacity)
	{
		current_float_t ideal_term = (heat_capacity - 1) * density * (std::max)(energy, grav_eq_utils::epsilon);
		current_float_t polytropic = 0.1f * std::pow(std::abs(density), polytropic_coef);
		return (std::max)(ideal_term + polytropic, grav_eq_utils::epsilon);
	}

	inline static bool is_beyond_radius(const point& dist, current_float_t radius)
	{
		return (dist.get_norm2() > radius * radius);
	}

	struct iteration_result
	{
		point dV;
		current_float_t dE;
		current_float_t dR;
		int interactions_count;
		current_float_t dT_CFL;
	};

	inline iteration_result iterate_particle(
		node* particle_node,
		particle& current_prt,
		grav_eq_iteration_buffers& buffers,
		const current_float_t heat_capacity, const current_float_t polytropic_coef, const current_float_t time_step)
	{
		constexpr current_float_t error_edge_squared = 0.05f;
		constexpr current_float_t courant_number = 0.3f;
		constexpr bool is_complete_SPH = true;
		node* cur_node = current.root_node;
		int interactions_counter = 0;
		current_float_t cur_density = 0;
		current_float_t cur_energy = 0;
		current_float_t cur_pressure = 0;

		if constexpr (is_complete_SPH)
		{
			spatial_neighbors.collect_neighbors(
				particle_node,
				buffers.radial_nodes);
			cur_density = get_density_at(
				particle_node);
			cur_energy = get_energy_at(
				particle_node);
			cur_pressure = get_pressure(cur_density, cur_energy, polytropic_coef, heat_capacity);
		}

		point gravity = barnes_hutt_force_in_subtree(
			cur_node,
			current_prt,
			error_edge_squared,
			&buffers.gravity_traversal);

		current_float_t c_i = 0;
		if constexpr (is_complete_SPH)
			c_i = sqrt(polytropic_coef * (std::max)(cur_pressure, grav_eq_utils::epsilon) / (std::max)(cur_density, grav_eq_utils::epsilon));

		current_float_t dR = 0;
		current_float_t dE = 0;

		current_float_t max_mu = 0;
		current_float_t delta_time_CFL = 0;

		point dV = {0, 0};
		current_float_t nabla_velocity = 0;

		auto mu__old = [&](const particle& prt)
		{
			const point velocity_difference = (current_prt.velocity - prt.velocity);
			const point position_difference = (current_prt.position - prt.position);
			const current_float_t radius = (std::max)(current_prt.radius, prt.radius);
			current_float_t prod = velocity_difference * position_difference;
			current_float_t stabilizing_term = 0.02f;
			if (prod < 0)
				return radius * prod / (
					(position_difference.get_norm2() + stabilizing_term)
					);
			else
				return (current_float_t)0.f;
		};

		auto mu = [&](const particle& prt, current_float_t inner_node_pressure, current_float_t inner_node_density)
		{
			const point velocity_difference = (current_prt.velocity - prt.velocity);
			const point position_difference = (current_prt.position - prt.position);
			current_float_t h_ij = (current_prt.radius + prt.radius) / 2.0f;
			current_float_t eta2 = 0.02f * h_ij * h_ij;  // η = 0.1 h_ij
			current_float_t prod = velocity_difference * position_difference;
			if (prod < 0)
			{
				current_float_t mu_ij = h_ij * prod / (position_difference.get_norm2() + eta2);
				current_float_t c_i = sqrt(polytropic_coef * cur_pressure / cur_density);
				current_float_t c_j = sqrt(polytropic_coef * inner_node_pressure / inner_node_density);
				current_float_t c_ij = (c_i + c_j) / 2.0f;
				return (std::max)(mu_ij, -2.0f * c_ij);  // Cap |μ_ij| ≤ 2 * c_ij
			}
			else
			{
				return (current_float_t)0.0f;
			}
		};

		if (is_complete_SPH)
		{
			dR = current_prt.radius * 0.5f *
				(1.f + std::pow(
					(current_float_t)particle::desired_amount_of_interactions /
					(current_prt.interactions_count + 1),
					0.5f));
			dR = (std::max)(dR, grav_eq_utils::epsilon * __size * 0.1f);
			dR -= current_prt.radius;
		}

		//current_float_t max_Pi = 0;
		for (auto& it_node : buffers.radial_nodes)
		{
			if (!is_complete_SPH)
				break;
			if (it_node == particle_node)
				continue;

			auto pos_difference = current_prt.position - it_node->mass_center.position;
			auto vel_difference = current_prt.velocity - it_node->mass_center.velocity;
			auto avg_radius = (current_prt.radius + it_node->mass_center.radius) / 2.0f;

			if (is_beyond_radius(pos_difference, avg_radius))
				continue;

			auto inner_node_density = get_density_at(
				it_node);
			auto inner_node_energy = get_energy_at(
				it_node);
			auto inner_node_pressure = get_pressure(inner_node_density, inner_node_energy, polytropic_coef, heat_capacity);
			auto core_gradient = grav_eq_utils::pressure_core_gradient(pos_difference, avg_radius);

			current_float_t rho_ij = (cur_density + inner_node_density) / 2.0f;
			current_float_t mu_val = mu(it_node->mass_center, inner_node_pressure, inner_node_density);
			current_float_t c_j = sqrt(polytropic_coef * (std::max)(inner_node_pressure, grav_eq_utils::epsilon) / (std::max)(inner_node_density, grav_eq_utils::epsilon));
			current_float_t c_ij = (c_i + c_j) / 2.0f;
			current_float_t Pi_ij = 0.0f;

			if (mu_val < 0 && std::abs(rho_ij) > grav_eq_utils::epsilon)
			{
				Pi_ij = (-0.5f * c_ij * mu_val + 1.0f * mu_val * mu_val) / std::abs(rho_ij);  // β=1.0f (reduced)
				// Pi_ij = (std::min)(Pi_ij, 10.0f * c_ij * c_ij / rho_ij);  // Cap Π_ij
				max_mu = (std::max)(max_mu, -mu_val);
			}

			if (Pi_ij > 1)
				Pi_ij = 1.f; // Hard cap Π_ij

			if (cur_pressure < 0 || inner_node_pressure < 0)
			{
				printf("Negative pressure detected: cur_P=%f, inner_P=%f, cur_rho=%f, inner_rho=%f, cur_u=%f, inner_u=%f\n",
					cur_pressure, inner_node_pressure, cur_density, inner_node_density, current_prt.energy, it_node->mass_center.energy);
			}

			nabla_velocity +=
				it_node->mass_center.mass * vel_difference * core_gradient;

			constexpr current_float_t Pi_coef = 1.f;

			dV += it_node->mass_center.mass * (
				inner_node_pressure / (inner_node_density * inner_node_density) +
				cur_pressure / (cur_density * cur_density) +
				Pi_coef * Pi_ij
				) * core_gradient;

			dE +=
				it_node->mass_center.mass * vel_difference * (
					inner_node_pressure / (inner_node_density * inner_node_density) +
					cur_pressure / (cur_density * cur_density) +
					Pi_coef * 0.5f * Pi_ij
					) * core_gradient;

			//max_Pi = (std::max)(max_Pi, Pi_ij);
			interactions_counter++;
		}

		// std::cout << max_Pi << std::endl;

		if constexpr (is_complete_SPH)
			nabla_velocity = -nabla_velocity / (std::max)(cur_density, grav_eq_utils::epsilon);
		/*delta_time_CFL = (std::min)(sqrt(current_prt.radius / dV.get_norm()),
			(std::min)(courant_number * current_prt.radius / (current_prt.velocity.get_norm()),
				abs(courant_number * current_prt.radius /
			(current_prt.radius * std::abs(nabla_velocity) + cur_energy + 1.2f * (cur_energy + 0.5f * max_mu)))
		));*/

		const point total_acceleration = -dV + gravity;
		const current_float_t resolved_length = (std::max)(
			current_prt.radius,
			grav_eq_utils::epsilon * __size * 0.1f);
		const current_float_t gravity_time_step = courant_number * sqrt(
			resolved_length /
			(std::max)(total_acceleration.get_norm(), grav_eq_utils::epsilon));
		const current_float_t crossing_time_step =
			courant_number * resolved_length /
			(std::max)(current_prt.velocity.get_norm(), grav_eq_utils::epsilon);
		delta_time_CFL = (std::min)(gravity_time_step, crossing_time_step);

		if (is_complete_SPH)
			delta_time_CFL = (std::min)(delta_time_CFL, (std::min)(
				sqrt(current_prt.radius / (std::max)(dV.get_norm(), grav_eq_utils::epsilon)),
				(std::min)(
					courant_number * current_prt.radius / (std::max)(current_prt.velocity.get_norm(), grav_eq_utils::epsilon),
					courant_number * current_prt.radius / (std::max)(
						current_prt.radius * abs(nabla_velocity) + c_i + 1.2f * (c_i + max_mu),
						grav_eq_utils::epsilon
						)
					)
				));

		//dE *= (polytropic_coef - 1) / std::pow(std::abs(cur_density), polytropic_coef - 1) * (cur_density > 0 ? 1 : -1);

		return {total_acceleration, dE, dR, interactions_counter, delta_time_CFL};
	}

	inline particle iterate_over_particle(
		node* particle_node,
		grav_eq_iteration_buffers& buffers,
		const current_float_t heat_capacity, const current_float_t polytropic_coef, const current_float_t time_step)
	{
		particle& current_prt = particle_node->mass_center;
		particle local_prt = current_prt;
		const auto ans = iterate_particle(
			particle_node,
			local_prt,
			buffers,
			heat_capacity, polytropic_coef, time_step);

		local_prt.energy += time_step * ans.dE;
		local_prt.interactions_count = ans.interactions_count;
		local_prt.radius = (std::max)(
			local_prt.radius + ans.dR,
			grav_eq_utils::epsilon * __size * 0.1f);
		local_prt.acceleration = ans.dV;

		// Symplectic Euler: all particles are accelerated from the same tree state,
		// then drift exactly once. This is first order but bounded for orbital motion.
		local_prt.velocity += time_step * local_prt.acceleration;
		local_prt.position += time_step * local_prt.velocity;

#ifdef is_variable_timestep
		local_prt.cfl_time = ans.dT_CFL;
#endif
		return local_prt;
	}

	inline void iterate_subtree(
		node* subtree_root,
		grav_eq_iteration_buffers& buffers)
	{
		buffers.subtree_traversal.clear();
		node* cur_node = subtree_root;
		node** ptemp;
		while (true)
		{
			if (cur_node)
			{
				if (cur_node->particles_count_in_subtrees)
				{
					for (node::positioning i = node::positioning::leftbottom; i < node::positioning::null; ((int&)i)++)
					{
						if (*(ptemp = cur_node->get_dptr(i)))
						{
							buffers.subtree_traversal.push_back(*ptemp);
						}
					}
				}
				else
				{
					auto prt = iterate_over_particle(
						cur_node,
						buffers,
						heat_capacity,
						polytropic_coef,
						local_time_step);
					cur_node->mass_center.visited = flickering;
					const bool finite_particle =
						std::isfinite(prt.position[0]) && std::isfinite(prt.position[1]) &&
						std::isfinite(prt.velocity[0]) && std::isfinite(prt.velocity[1]) &&
						std::isfinite(prt.acceleration[0]) && std::isfinite(prt.acceleration[1]) &&
						std::isfinite(prt.mass) && std::isfinite(prt.radius) &&
						std::isfinite(prt.energy)
#ifdef is_variable_timestep
						&& std::isfinite(prt.cfl_time)
#endif
						;
					if (finite_particle)
					{
						if (!grav_eq_utils::point_in_square(buffer.root_node->leftbottom_corner, buffer.root_node->righttop_corner, prt.position))
						{
							float x_min = buffer.root_node->leftbottom_corner[0];
							float x_max = buffer.root_node->righttop_corner[0];
							float y_min = buffer.root_node->leftbottom_corner[1];
							float y_max = buffer.root_node->righttop_corner[1];
							float x_range = x_max - x_min;
							float y_range = y_max - y_min;

							auto periodic_wrap = [](current_float_t value, current_float_t min_value, current_float_t range)
							{
								current_float_t wrapped = std::fmod(value - min_value, range);
								if (wrapped < 0)
									wrapped += range;
								return min_value + wrapped;
							};
							prt.position[0] = periodic_wrap(prt.position[0], x_min, x_range);
							prt.position[1] = periodic_wrap(prt.position[1], y_min, y_range);
						}

						buffer_mutex.lock();
						buffer.push(prt);
						buffer_mutex.unlock();
					}
					else
						printf("non-finite particle state rejected\n");
				}
			}
			if (cur_node && !buffers.subtree_traversal.empty())
			{
				cur_node->mass_center.visited = flickering;
				cur_node = buffers.subtree_traversal.back();
				buffers.subtree_traversal.pop_back();
			}
			else
				break;
		}
	}

	inline static size_t subtree_particle_count(const node* subtree_root)
	{
		if (!subtree_root)
			return 0;
		if (!subtree_root->particles_count_in_subtrees &&
			std::abs(subtree_root->mass_center.mass) <= grav_eq_utils::epsilon)
			return 0;
		return static_cast<size_t>(subtree_root->particles_count_in_subtrees) + 1;
	}

	inline void build_subdivision_tasks()
	{
		const size_t total_particles = subtree_particle_count(current.root_node);
		_subdivision_roots.clear();
		_next_subdivision_root.store(0, std::memory_order_relaxed);
		if (!total_particles)
			return;

		constexpr size_t tasks_per_worker = 8;
		const size_t target_task_count = (std::min)(
			total_particles,
			num_of_threads * tasks_per_worker);
		using weighted_root = std::pair<size_t, node*>;
		std::priority_queue<weighted_root> frontier;
		frontier.push({total_particles, current.root_node});

		while (frontier.size() < target_task_count)
		{
			const auto [particle_count, subtree_root] = frontier.top();
			frontier.pop();

			if (!subtree_root->particles_count_in_subtrees)
			{
				frontier.push({particle_count, subtree_root});
				break;
			}

			size_t child_count = 0;
			for (int position = node::positioning::leftbottom;
				position < node::positioning::null;
				position++)
			{
				node* child = subtree_root->get(
					static_cast<node::positioning>(position));
				const size_t child_particles = subtree_particle_count(child);
				if (child_particles)
				{
					frontier.push({child_particles, child});
					child_count++;
				}
			}

			if (!child_count)
			{
				frontier.push({particle_count, subtree_root});
				break;
			}
		}

		_subdivision_roots.reserve(frontier.size());
		while (!frontier.empty())
		{
			_subdivision_roots.push_back(frontier.top().second);
			frontier.pop();
		}
	}

	inline void prepare_iteration()
	{
		spatial_neighbors.build_neighbor_graph(current.root_node);
		build_subdivision_tasks();
	}

	inline void subdivide_tree()
	{
		prepare_iteration();

		auto now = std::chrono::steady_clock::now();

#ifndef measuring_performance
		auto difference = std::chrono::duration_cast<std::chrono::seconds>(now - last_measure).count();
		if (difference > 2)
			if (difference > 2)
			{
				last_measure = now;
				printf("%i particles\n", current.root_node->particles_count_in_subtrees);

#ifdef is_variable_timestep
				printf("cfl_time: %.10lf; total_time: %lf\n", current.root_node->mass_center.cfl_time, total_time);
#else
				printf("time_step: %.10lf; total_time: %lf\n", time_step, total_time);
#endif			
			}
#else
		printf("Delta time: %lf\n", difference.count());
#endif

#ifdef is_variable_timestep
		local_time_step = (std::max)((std::min)(current.root_node->mass_center.cfl_time, time_step), (current_float_t)1e-10f);
#else
		local_time_step = time_step;
#endif
		total_time += local_time_step;

	}

	inline void start_threads()
	{
		if (threads.size())
			return;

		subdivide_tree();
		for (int i = 0; i < num_of_threads; i++)
		{
			threads.push_back(new pooled_thread()); // executors
			auto t = threads.back()->__void_ptr_accsess();
			auto* info = new worker_thread_info;
			info->id = i;
			*t = info;
			threads.back()->set_new_function([this](void** void_ptr)
			{
				auto* info = static_cast<worker_thread_info*>(*void_ptr);

				pause.lock();
				pause.unlock();

				while (true)
				{
					const size_t task_index = _next_subdivision_root.fetch_add(
						1,
						std::memory_order_relaxed);
					if (task_index >= _subdivision_roots.size())
						break;
					iterate_subtree(
						_subdivision_roots[task_index],
						info->buffers);
				}

				//printf("thread finished\n");

			});
		}

		threads.push_back(new pooled_thread());//observer
		threads.back()->set_new_awaiting_time(10);
		threads.back()->set_new_default_state(pooled_thread::state::waiting);
		threads.back()->set_new_function([this](void** ptr)
		{
			for (auto ptr : threads)
				if (ptr != threads.back() && ptr->get_state() != pooled_thread::state::idle)
				{
					threads.back()->sign_awaiting();
					return;
				}

			//printf("not subdivided\n");

			pause.lock();
			pause.unlock();
			pre_swap.lock();
			current.clear();
			current.swap(buffer);
			// The sampled renderer reads the neighbour grid as well as the
			// current tree. Publish both under the same lock so a frame cannot
			// observe a new tree paired with a half-built sampling index.
			subdivide_tree();
			pre_swap.unlock();

			for (auto ptr : threads)
				ptr->sign_awaiting();

			//printf("subdivided\n");

		});
		for (size_t worker_index = 0; worker_index < num_of_threads; worker_index++)
			threads[worker_index]->sign_awaiting();
		threads.back()->sign_awaiting();
	}
};

#pragma once

#include "consts.h"
#include "buffered_queue_spsc.h"
#include "field_vis.h"
#include "sq_matrix.h"
#include "weird_hacks.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <complex>
#include <functional>
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
inline bool square_n_circle_intersection(const point& lb_sq, const point& rt_sq, const point& c_cen_pos, current_float_t radius)
{
	// clamp(value, min, max) - limits value to the range min..max
	point closest = {clamp(_x(c_cen_pos), _x(lb_sq), _x(rt_sq)) , clamp(_y(c_cen_pos), _y(lb_sq), _y(rt_sq))};
	point difference = c_cen_pos - closest;
	return difference.get_norm2() < radius * radius;
}
inline bool point_in_square(const point& lb_sq, const point& rt_sq, const point& p_pos)
{
	point center = (rt_sq + lb_sq) * 0.5f;
	current_float_t width = (_x(rt_sq) - _x(lb_sq)) * 0.5f;
	return (abs(_x(center) - _x(p_pos)) < width) && (abs(_y(center) - _y(p_pos)) < width);
}
inline bool point_in_circle(const point& c_pos, current_float_t radius, const point& p_pos)
{
	return (c_pos - p_pos).get_norm() < radius;
}
inline bool circle_inside_square(const point& lb_sq, const point& rt_sq, const point& c_r_pos, current_float_t radius)
{
	//point center = (rt_sq + lb_sq) * 0.5f;
	point vec = c_r_pos - (rt_sq + lb_sq) * 0.5f;
	point n_vec = {(vec[0] >= 0) ? 1 : -1, (vec[1] >= 0) ? 1 : -1};
	n_vec *= radius;
	return n_vec >= lb_sq && n_vec <= rt_sq;
}
constexpr current_float_t epsilon = 5e-3;
inline current_float_t pressure_core(const point& r, current_float_t h)
{
	constexpr current_float_t constant = 4.;
	current_float_t r_norm = r.get_norm();
	if (r_norm < h)
		return constant * std::pow(h - r_norm, 3) / std::pow(h, 4);
	else
		return 0.;
}
inline point pressure_core_gradient(const point& r, current_float_t h)
{
	constexpr current_float_t constant = -12.f;
	current_float_t r_norm = r.get_norm();
	if (r_norm < h && std::abs(r_norm) > epsilon)
		return constant * (r / r_norm) * std::pow(h - r_norm, 2) / std::pow(h, 4);
	else
		return {0,0};
}
inline current_float_t pressure_core(current_float_t r, current_float_t h)
{
	constexpr current_float_t constant = 4.;
	if (r < h)
		return constant * std::pow(h - r, 3) / std::pow(h, 4);
	else
		return 0.;
}
inline current_float_t inverse_pressure_core(current_float_t d, current_float_t h)
{
	if (0 <= d && d <= h)
		return h - std::pow(h * h * h * h * d / 4., 1. / 3.);
	else return 0;
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
	static constexpr int desired_amount_of_interactions = 10;//25^(2/3) ~ 8.5f//+1 for noninteractive particle
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
		return (pos >= leftbottom_corner && pos <= righttop_corner);
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
	vecnode first_corad;
	vecnode second_corad;
	vecnode gravity_traversal;
	vecnode radius_traversal;
};

//some time before it was an object...
//not more than O(logN) in case of *not specifically built tree*
inline void find_nodes_in_radius(
	node* center,
	current_float_t radius,
	vecnode& results,
	vecnode& traversal,
	const point* source_override = nullptr)
{
	const point source = source_override ? *source_override : center->mass_center.position;
	while (center->parent && !grav_eq_utils::circle_inside_square(center->leftbottom_corner, center->righttop_corner, source, radius)) //deriving from old style RNC
		center = center->parent;
	results.clear();
	traversal.clear();
	traversal.reserve(center->particles_count_in_subtrees + 1);
	node* cur_node = center;
	node** ptemp = &center;
	while (true)
	{
		if (cur_node)
		{
			if (cur_node->particles_count_in_subtrees)
			{
				for (node::positioning i = node::positioning::leftbottom; i < node::positioning::null; ((int&)i)++)
				{
					if (*(ptemp = cur_node->get_dptr(i)) &&
						grav_eq_utils::square_n_circle_intersection((*ptemp)->leftbottom_corner, (*ptemp)->righttop_corner, source, radius))
						traversal.push_back(*ptemp);
				}
			}
			else if (std::abs(cur_node->mass_center.mass) > grav_eq_utils::epsilon &&
				grav_eq_utils::point_in_circle(source, radius, cur_node->mass_center.position))
			{
				results.push_back(cur_node);
			}
		}
		if (!traversal.empty())
		{
			cur_node = traversal.back();
			traversal.pop_back();
		}
		else
			break;
	}
	//printf("radius_nodes: %i\n", rad_nodes->size());
}

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
					if (extended_draw)
					{
						draw_smooth_circle(_x(position), _y(position),
							cur_node.first->mass_center.radius * relative_size,
							particle_value * value_decrimemnt,
							1.10f, 15
						);
					}
					else
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

	inline static current_float_t get_density_at(
		node* begin,
		vecnode& reserved_rad_nodes,
		vecnode& radius_traversal,
		particle* rsv_part = nullptr)
	{
		particle source = (rsv_part) ? *rsv_part : begin->mass_center;
		find_nodes_in_radius(
			begin,
			source.radius,
			reserved_rad_nodes,
			radius_traversal,
			rsv_part ? &rsv_part->position : nullptr);
		current_float_t sum = 0;
		for (auto& cur_node : reserved_rad_nodes)
		{
			auto pos_difference = source.position - cur_node->mass_center.position;
			auto max_radius = (std::max)(source.radius, cur_node->mass_center.radius);
			if (is_beyond_radius(pos_difference, max_radius))
				continue;
			sum += cur_node->mass_center.mass * grav_eq_utils::pressure_core(pos_difference, max_radius);
		}
		return sum;
	}

	inline static current_float_t get_energy_at(
		node* begin,
		vecnode& reserved_rad_nodes,
		vecnode& reserved_drn,
		vecnode& radius_traversal,
		particle* rsv_part = nullptr)
	{
		particle source = (rsv_part) ? *rsv_part : begin->mass_center;
		find_nodes_in_radius(
			begin,
			source.radius,
			reserved_rad_nodes,
			radius_traversal,
			rsv_part ? &rsv_part->position : nullptr);
		current_float_t sum = 0;
		for (auto& cur_node : reserved_rad_nodes)
		{
			auto pos_difference = source.position - cur_node->mass_center.position;
			auto max_radius = (std::max)(source.radius, cur_node->mass_center.radius);
			if (is_beyond_radius(pos_difference, max_radius))
				continue;
			sum +=
				(cur_node->mass_center.mass /
					get_density_at(cur_node, reserved_drn, radius_traversal))
				* cur_node->mass_center.energy * grav_eq_utils::pressure_core(pos_difference, max_radius);
		}
		return sum;
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
		particle& current_prt,
		grav_eq_iteration_buffers& buffers,
		const current_float_t heat_capacity, const current_float_t polytropic_coef, const current_float_t time_step)
	{
		constexpr current_float_t error_edge_squared = 0.05f;
		constexpr current_float_t courant_number = 0.3f;
		constexpr bool is_complete_SPH = true;
		node* cur_node = current.root_node;
		int interactions_counter = 0;
		buffers.first_corad.clear();
		buffers.second_corad.clear();
		current_float_t cur_density = 0;
		current_float_t cur_energy = 0;
		current_float_t cur_pressure = 0;

		if constexpr (is_complete_SPH)
		{
			find_nodes_in_radius(
				cur_node,
				current_prt.radius,
				buffers.radial_nodes,
				buffers.radius_traversal,
				&current_prt.position);
			cur_density = get_density_at(
				cur_node,
				buffers.first_corad,
				buffers.radius_traversal,
				&current_prt);
			cur_energy = get_energy_at(
				cur_node,
				buffers.first_corad,
				buffers.second_corad,
				buffers.radius_traversal,
				&current_prt);
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
					0.33333f));
			dR = (std::max)(dR, grav_eq_utils::epsilon * __size * 0.1f);
			dR -= current_prt.radius;
		}

		//current_float_t max_Pi = 0;
		for (auto& it_node : buffers.radial_nodes)
		{
			if (!is_complete_SPH)
				break;

			auto pos_difference = current_prt.position - it_node->mass_center.position;
			auto vel_difference = current_prt.velocity - it_node->mass_center.velocity;
			auto avg_radius = (current_prt.radius + it_node->mass_center.radius) / 2.0f;

			if (is_beyond_radius(pos_difference, avg_radius) || pos_difference.get_norm2() < grav_eq_utils::epsilon)
				continue;

			auto inner_node_density = get_density_at(
				it_node,
				buffers.first_corad,
				buffers.radius_traversal);
			auto inner_node_energy = get_energy_at(
				it_node,
				buffers.first_corad,
				buffers.second_corad,
				buffers.radius_traversal);
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
		particle& current_prt,
		grav_eq_iteration_buffers& buffers,
		const current_float_t heat_capacity, const current_float_t polytropic_coef, const current_float_t time_step)
	{
		particle local_prt = current_prt;
		const auto ans = iterate_particle(
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
						cur_node->mass_center,
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

	inline void subdivide_tree()
	{
		build_subdivision_tasks();

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
			pre_swap.unlock();

			subdivide_tree();

			for (auto ptr : threads)
				ptr->sign_awaiting();

			//printf("subdivided\n");

		});
		for (size_t worker_index = 0; worker_index < num_of_threads; worker_index++)
			threads[worker_index]->sign_awaiting();
		threads.back()->sign_awaiting();
	}
};

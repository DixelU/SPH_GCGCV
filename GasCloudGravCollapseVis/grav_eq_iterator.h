#pragma once
#include <cmath>
#include <thread>
#include <complex>
#include <vector>
#include <functional>
#include <atomic>
#include "field_vis.h"
#include "weird_hacks.h"
#include "consts.h"
#include "multidimentional_point.h"

#include <stack>
#include <queue>

#include "allocator.h"

using namespace std;
using point = Point<2>;

namespace _____type_desc {
	auto p_zero = [&](double t, const point& x) -> point { return point(); };
	auto d_zero = [&](double t, double x) -> double { return 0.; };
}

using p_lambda = std::function<point(double, const point&)>;
using d_lambda = std::function<double(double, double)>;

namespace grav_eq_utils {
	inline bool square_n_circle_intersection(const point& lb_sq, const point& rt_sq, const point& c_cen_pos, double radius)	{
		// clamp(value, min, max) - limits value to the range min..max
		point closest = { clamp(_x(c_cen_pos), _x(lb_sq), _x(rt_sq)) , clamp(_y(c_cen_pos), _y(lb_sq), _y(rt_sq)) };
		point difference = c_cen_pos - closest;
		return difference.norma2() < radius;
	}
	inline bool point_in_square(const point& lb_sq, const point& rt_sq, const point& p_pos) {
		point center = (rt_sq + lb_sq) * 0.5;
		double width = (_x(rt_sq) - _x(lb_sq)) * 0.5;
		return (abs(_x(center) - _x(p_pos)) < width) && (abs(_y(center) - _y(p_pos)) < width);
	}
	inline bool point_in_circle(const point& c_pos, double radius, const point& p_pos) {
		return (c_pos - p_pos).norma() < radius;
	}
	inline bool circle_inside_square(const point& lb_sq, const point& rt_sq, const point& c_r_pos, double radius) {
		//point center = (rt_sq + lb_sq) * 0.5;
		point vec = c_r_pos - (rt_sq + lb_sq) * 0.5;
		point n_vec = { (vec[0] >= 0) ? 1 : -1, (vec[1] >= 0) ? 1 : -1 };
		n_vec *= radius;
		return n_vec >= lb_sq && n_vec <= rt_sq;
	}
	constexpr double epsilon = 0.005;
	inline double pressure_core(const point& r, double h) {
		constexpr double constant = 4.;
		double r_norm = r.norma();
		if (r_norm < h) 
			return constant* std::pow(h - r_norm, 3) / std::pow(h, 4);
		else
			return 0.;
	}
	inline point pressure_core_gradient(const point& r, double h) {
		constexpr double constant = -12.;
		double r_norm = r.norma();
		if (r_norm < h && std::abs(r_norm)>epsilon)
			return constant * (r / r_norm) * std::pow(h - r_norm, 2) / std::pow(h, 4);
		else
			return {0,0};
	}
	inline double pressure_core(double r, double h) {
		constexpr double constant = 4.;
		if (r < h)
			return constant * std::pow(h - r, 3) / std::pow(h, 4);
		else
			return 0.;
	}
	inline double inverse_pressure_core(double d, double h) {
		if (0 <= d && d <= h)
			return h - std::pow(h * h * h * h * d / 4, 1. / 3);
		else return 0;
	}
};

inline void draw_smooth_circle(const float x, const float y, const float r, const float value, const float dvalue, const float dangle) {
	float begin = grav_eq_utils::pressure_core(0, r);
	for (float val = begin; val > 0.005; val /= dvalue) {

		float rad = grav_eq_utils::inverse_pressure_core(val, r);

		auto [pr, pg, pb] = get_color(val*value);

		glColor4f(pr, pg, pb, 0.025f);

		glBegin(GL_POLYGON);
	
		for (float i = 0.f; i < 360.f; i += dangle)
			glVertex2f(rad * cos(ANGTORAD(i)) + x, rad * sin(ANGTORAD(i)) + y);

		glEnd();

	}
}

struct particle {
	static constexpr int desired_amount_of_interactions = 10;//25^(2/3) ~ 8.5//+1 for noninteractive particle
	int interactions_count;
	point position;
	point velocity;
	point acceleration;
	double mass;
	double radius;
	double energy;
	double cfl_time;
	bool visited;
	particle(point position = { 0.,0. }, point velocity = { 0.,0. }, point acceleration = { 0.,0. }, double part_mass = 0., double radius = 0., double energy = 0., int amount_of_interactions = 1, double cfl_time = 1) :
		position(position), velocity(velocity), mass(part_mass), radius(radius), energy(energy), acceleration(acceleration), interactions_count(amount_of_interactions), cfl_time(cfl_time) {
		visited = false;
	}
	inline bool operator==(const particle& prt) const {
		using namespace grav_eq_utils;
		return (position - prt.position).norma() < epsilon && (velocity - prt.velocity).norma() < epsilon;
	}
	//inverse to operator-
	inline particle operator+(const particle& prt) const {
		double ratio = mass / (prt.mass + mass);
		double aratio = 1. - ratio;
		return particle(
			ratio * position + aratio * prt.position,
			ratio * velocity + aratio * prt.velocity,
			ratio * acceleration + aratio * prt.acceleration,
			prt.mass + mass,
			sqrt(radius* radius + prt.radius* prt.radius),
			energy + prt.energy,
			interactions_count + prt.interactions_count,
			min(cfl_time, prt.cfl_time)
		);
	}
	//inverse to operator+
	inline particle operator-(const particle& prt) const {
		double ratio = mass / (mass - prt.mass);
		return particle(
			ratio * (position - prt.position) + prt.position,
			ratio * (velocity - prt.velocity) + prt.velocity,
			ratio * (acceleration - prt.acceleration) + prt.acceleration,
			mass - prt.mass,
			sqrt(radius * radius - prt.radius * prt.radius),
			energy - prt.energy,
			interactions_count - prt.interactions_count,
			min(cfl_time, prt.cfl_time)
		);
	}
	inline particle operator+=(const particle& prt) {
		return ((*this) = (*this) + prt);
	}
	inline particle operator-=(const particle& prt) {
		return ((*this) = (*this) - prt);
	}
};


namespace draw_type {
	enum class dt {
		density, energy, x_speed, y_speed, x_acceleration, y_acceleration
	};
}


struct node {
	enum positioning {
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
	node() {
		null_node = nullptr;
		left_bottom = left_top = right_bottom = right_top = parent = nullptr;
		particles_count_in_subtrees = 0;
		leftbottom_corner = righttop_corner = { 0,0 };
		mass_center = particle();
	}
	node(node* parent, positioning pos_id) :node() {
		this->parent = parent;
		point center = (parent->righttop_corner + parent->leftbottom_corner) / 2.;
		mass_center.position = center;
		point local_shift(vector<double>{ 0., _y(center - parent->leftbottom_corner) });
		switch (pos_id) {
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
	node(node* parent, const particle& p) : node(parent, get_positioning(parent, p.position)) {
		mass_center = p;
	}
	inline static positioning get_positioning(node* nd, const point& pos) {
		if (!nd || !nd->point_is_inside(pos))
			return null;
		point center = (nd->righttop_corner + nd->leftbottom_corner) / 2.;
		point difference = pos - center;
		if (_x(difference) >= 0) {
			if (_y(difference) >= 0)
				return righttop;
			else
				return rightbottom;
		}
		else {
			if (_y(difference) >= 0)
				return lefttop;
			else
				return leftbottom;
		}
	}
	inline node** get_dptr(positioning D) {
		switch (D) {
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
	inline bool point_is_inside(const point& pos) {
		return (pos >= leftbottom_corner && pos <= righttop_corner);
	}
	inline void zero_pointers() {
		null_node = nullptr;
		left_bottom = left_top = right_bottom = right_top = parent = nullptr;
	}
	inline node*& get(positioning D) {
		return *get_dptr(D);
	}
};


//some time before it was an object...
//not more than O(logN) in case of *not specifically built tree*
inline void radius_node_catcher(node* center, double radius, std::vector<node*>* rad_nodes, point* rsv_source = nullptr) {
	point source = (rsv_source)? *rsv_source:center->mass_center.position;
	while (center->parent && !grav_eq_utils::circle_inside_square(center->leftbottom_corner, center->righttop_corner, source, radius)) //deriving from old style RNC
		center = center->parent;
	rad_nodes->clear();
	std::stack<node*> cur_nodes;
	node* cur_node = center;
	node** ptemp = &center;
	while (true) {
		if (cur_node) {
			if (cur_node->particles_count_in_subtrees ) {
				for (node::positioning i = node::positioning::leftbottom; i < node::positioning::null; ((int&)i)++) {
					if ( *(ptemp = cur_node->get_dptr(i)) &&
						grav_eq_utils::square_n_circle_intersection((*ptemp)->leftbottom_corner, (*ptemp)->righttop_corner, source, radius)) 
						cur_nodes.push(*ptemp);
				}
			}
			else if(std::abs(cur_node->mass_center.mass) > grav_eq_utils::epsilon && grav_eq_utils::point_in_circle(source, radius, cur_node->mass_center.position)) {
				rad_nodes->push_back(cur_node);
			}
		}
		if (cur_nodes.size()) {
			cur_node = cur_nodes.top();
			cur_nodes.pop();
		}
		else
			break;
	}
	//printf("radius_nodes: %i\n", rad_nodes->size());
}

struct quad_tree {
	//node*, node* = (temp,cur_node)
	using positioning = node::positioning;

	node* root_node;
	//node_reserver* reserver;
	recursive_mutex locker;
	recursive_mutex swap_prevention;
	quad_tree() /*: reserver(new node_reserver())*/ {
		root_node = nullptr;
	}
	quad_tree(double size) : quad_tree() {
		//root_node = reserver->get_unused();
		root_node = new node();
		root_node->leftbottom_corner = { -size * 0.5,-size * 0.5 };
		root_node->righttop_corner = { size * 0.5,size * 0.5 };
	}
	~quad_tree() {
		clear();
	}

	inline void clear() {
		locker.lock();
		std::stack<node*> cur_nodes;

		node* cur_node = root_node;
		node* temp = nullptr;
		node** ptemp;
		for (positioning i = positioning::leftbottom; i < positioning::null; ((int&)i)++) {
			if (*(ptemp = root_node->get_dptr(i))) {
				cur_nodes.push(*ptemp);
				*ptemp = nullptr;
			}
		}
		if (cur_nodes.size()) {
			cur_node = cur_nodes.top();
			cur_nodes.pop();
		}
		else
			return;
		while (true) {
			if (cur_node) {
				if (cur_node->particles_count_in_subtrees) {
					for (positioning i = positioning::leftbottom; i < positioning::null; ((int&)i)++) {
						if (*(ptemp = cur_node->get_dptr(i))) {
							cur_nodes.push(*ptemp);
						}
					}
				}
			}
			if (cur_nodes.size()) {
				//cur_node->zero_pointers();
				delete cur_node;
				cur_node = cur_nodes.top();
				cur_nodes.pop();
			}
			else
				break;
		}
		root_node->particles_count_in_subtrees = 0;
		root_node->mass_center = particle();
		locker.unlock();
	}

	inline void swap(quad_tree &tree) {
		swap_prevention.lock();
		tree.swap_prevention.lock();
		locker.lock();
		tree.locker.lock();

		std::swap(root_node,tree.root_node);
		
		tree.locker.unlock();
		locker.unlock();
		tree.swap_prevention.unlock();
		swap_prevention.unlock();
	}

	inline node** push(const particle& prt, unsigned char level = 0) {
		constexpr unsigned char max_level = 50;
		using namespace grav_eq_utils;
		node::positioning prt_pos = node::positioning::null; 
		node** temp = nullptr;
		locker.lock();
		node* nd = root_node;
	prp_begining:
		if (!nd || !nd->point_is_inside(prt.position)) 
			goto prp_ending;

		if (std::abs(nd->mass_center.mass) <= epsilon || 
			(nd->mass_center.position - prt.position).norma2() < epsilon * epsilon || 
			level >= max_level) {
			nd->mass_center += prt;
			goto prp_ending;
		}

		if (!nd->particles_count_in_subtrees) {
			node::positioning mc_pos = node::get_positioning(nd, nd->mass_center.position);
			node** temp = nd->get_dptr(mc_pos);
			if (!*temp) {
				*temp = new node(nd, mc_pos);
				(*temp)->mass_center = nd->mass_center;
				nd->particles_count_in_subtrees++;
			}
		}

		nd->mass_center += prt;
		nd->particles_count_in_subtrees++;
		prt_pos = node::get_positioning(nd, prt.position);
		temp = nd->get_dptr(prt_pos);
		if (!*temp) 
			*temp = new node(nd, prt_pos);
		nd = *temp;
		level++;
		goto prp_begining;
	prp_ending:
		locker.unlock();
		return temp;
	}

	inline void draw(int draw_level, const point& center, double side_size, float points_size, float value_decrimemnt, draw_type::dt type = draw_type::dt::density, bool extra_flare = false, bool edge_drawer = false, bool draw_points = false, bool extended_draw = false) {
		swap_prevention.lock();
		std::stack <pair<node*,int>> cur_nodes;
		point	lb_sc{ (0 - RANGE) * (WindX / WINDXSIZE) - centx, (0 - RANGE) * (WindY / WINDYSIZE) - centy}, 
				rt_sc{ { ( RANGE) * (WindX / WINDXSIZE) - centx, (RANGE) * (WindY / WINDYSIZE) - centy} };
		pair<node*, int> cur_node = { root_node , 0 };
		node* temp = nullptr;
		double size = (_x(root_node->righttop_corner) - _x(root_node->leftbottom_corner));
		side_size /= size;
		while (true) {
			if (cur_node.first) {
				if (cur_node.second<draw_level && cur_node.first->particles_count_in_subtrees 
					// && cur_node.first->righttop_corner >= lb_sc && cur_node.first->leftbottom_corner <= rt_sc // positioning on screen
					) {
					for (positioning i = positioning::leftbottom; i < positioning::null; ((int&)i)++) {
						if ((temp = cur_node.first->get(i))) {
							cur_nodes.push({ temp, cur_node.second + 1 });
						}
					}
				}
				else {
					point lb = (cur_node.first->leftbottom_corner * side_size + center);
					point rt = (cur_node.first->righttop_corner * side_size + center);
					double particle_value = 0;
					double node_value = 0;
					double ratio = (cur_node.first->mass_center.radius * cur_node.first->mass_center.radius) / std::pow(_x(cur_node.first->leftbottom_corner - cur_node.first->righttop_corner), 2);
					bool visited = cur_node.first->mass_center.visited;
					auto position = (cur_node.first->mass_center.position * side_size + center);
					switch (type) {
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

					if(edge_drawer){
						glBegin(GL_LINE_LOOP);
						glColor4f(nr, ng, nb, 1);
						glVertex2f(_x(lb), _y(lb));
						glVertex2f(_x(lb), _y(rt));
						glVertex2f(_x(rt), _y(rt));
						glVertex2f(_x(rt), _y(lb));
						glEnd();
					}
					if(extended_draw){
						draw_smooth_circle(_x(position), _y(position), 
							cur_node.first->mass_center.radius*side_size, 
							particle_value * value_decrimemnt,
							1.10, 60
						);
					}
					else {
						auto [pr, pg, pb] = get_color(particle_value * value_decrimemnt);
						auto a = (pr + pg + pb) * 0.15;

						glPointSize(side_size * 2 * cur_node.first->mass_center.radius);
						glColor4f(pr, pg, pb, 0.05 + 0.05 * visited + a);
						glBegin(GL_POINTS);
						glVertex2f(_x(position), _y(position));
						glEnd();
					}
					
					if (draw_points) {
						glPointSize(points_size + extra_flare * points_size);
						glColor4f(0.5 + extra_flare * visited, 0.5 - extra_flare * visited, 0.5 + 0.5 * visited, 0.5);
						glBegin(GL_POINTS);
						glVertex2f(_x(position), _y(position));
						glEnd();
					}
				}
			}
			if (cur_nodes.size()) {
				cur_node = cur_nodes.top();
				cur_nodes.pop();
			}
			else
				break;
		}
		swap_prevention.unlock();
	}
};

using vecnode = std::vector<node*>;

class pooled_thread {
public:
	enum class state {
		running, idle, waiting
	};
private:
	using funcT = std::function<void(void**)>;
	void* thread_data;//memory leak is allowed actually
	int await_in_milliseconds;
	funcT exec_func;
	bool is_active;
	mutable state cur_state;
	mutable state default_state;
	std::recursive_mutex execution_locker;
	void start_thread() {
		std::thread th([this]() {
			while (is_active) {
				execution_locker.lock();
				if (cur_state == state::waiting) {
					cur_state = state::running;
					exec_func(&thread_data);
				}
				cur_state = default_state;
				execution_locker.unlock();
				std::this_thread::sleep_for(std::chrono::milliseconds(await_in_milliseconds));
			}
			});
		th.detach();
	}
public:
	pooled_thread(funcT function = [](void** ptr) {return; }, int awaiting_time = 5) :exec_func(function),  await_in_milliseconds(awaiting_time), default_state(state::idle) {
		thread_data = nullptr;
		is_active = true;
		default_state = state::idle;
		start_thread();
	}
	~pooled_thread() {
		disable();
	}
	state get_state() const {
		return cur_state;
	}
	void sign_awaiting() {
		execution_locker.lock();
		cur_state = state::waiting;
		execution_locker.unlock();
	}
	void set_new_awaiting_time(int milliseconds) {
		execution_locker.lock();
		await_in_milliseconds = milliseconds;
		execution_locker.unlock();
	}
	void set_new_default_state(state def_state = state::idle) {
		execution_locker.lock();
		default_state = def_state;
		execution_locker.unlock();
	}
	void set_new_function(funcT func) {
		execution_locker.lock();
		exec_func = func;
		execution_locker.unlock();
	}
	void disable() {
		execution_locker.lock();
		is_active = false;
		execution_locker.unlock();
	}
	void** __void_ptr_accsess() {
		return &thread_data;
	}
};



struct grav_eq_processor {
	mutable std::vector<vecnode> threads_desired_roots;
	mutable std::stack <pair<node*, int>> _subdivision_cur_nodes;
	mutable std::vector<std::pair<int, node*>> _subdivision_roots;
	mutable std::vector<pooled_thread*> threads;
	const size_t num_of_threads;

	double heat_capacity;
	double polytropic_coef;
	double time_step;
	const double __size;
	quad_tree current, buffer;
	std::mutex buffer_mutex;
	std::mutex pre_swap;
	std::mutex pause;
	bool is_paused;
	bool flickering;
	bool reporting;
	bool halt_velocity;

	grav_eq_processor(const vector<particle>& input, double size);

	struct iteration_result {
		point dV;
		double dE;
		double dR;
		int interactions_count;
		double dT_CFL;
	};

	inline static double get_density_at(node* begin, vecnode& reserved_rad_nodes, particle* rsv_part = nullptr);
	inline static double get_energy_at(node* begin, vecnode& reserved_rad_nodes, vecnode& reserved_drn, particle* rsv_part = nullptr);
	inline static point grav_force(const particle& center, const particle& distant_prt);
	inline static point barnes_hutt_force_in_subtree(node* cur_node, const particle& current, const double error_edge_squared);
	inline static double get_pressure(double density, double energy, double polytropic_coef, double heat_capacity);
	inline static bool is_beyond_radius(const point& dist, double radius);

	inline iteration_result iterate_particle(particle& current_prt, vecnode* rad_vector, vecnode* corad_vector1, vecnode* corad_vector2,
		const double heat_capacity, const double polytropic_coef, const double time_step);
	inline particle iterate_over_particle(particle& current_prt, vecnode* rad_vector, vecnode* corad_vector1, vecnode* corad_vector2,
		const double heat_capacity, const double polytropic_coef, const double time_step);
	inline void iterate_subtree(node* subtree_root, std::stack<node*>* cur_nodes, vecnode* rad_nodes, vecnode* first_corad, vecnode* second_corad);
	inline void start_single_thread();
	inline void subdivide_tree();
	inline void start_threads();
};

struct grav_eq_processor {
	mutable std::vector<vecnode> threads_desired_roots;
	mutable std::stack <pair<node*, int>> _subdivision_cur_nodes;
	mutable std::vector<std::pair<int, node*>> _subdivision_roots;
	mutable std::vector<pooled_thread*> threads;
	const size_t num_of_threads;

	double heat_capacity;
	double polytropic_coef;
	double time_step;
	const double __size;
	quad_tree current, buffer;
	std::mutex buffer_mutex;
	std::mutex pre_swap;
	std::mutex pause;
	bool is_paused;
	bool flickering;
	bool reporting;
	bool halt_velocity;

	grav_eq_processor(const vector<particle>& input, double size) :
		current(size),
		buffer(size),
		heat_capacity(1.01),
		polytropic_coef(1.67),
		time_step(0.004), flickering(true), reporting(false), halt_velocity(false),
		num_of_threads(max(std::thread::hardware_concurrency() - 2, 1)),
		__size(size)
	{
		is_paused = false;

		for (int i = 0; i < num_of_threads; i++)
			threads_desired_roots.push_back(std::vector<node*>());

		for (auto& prt : input) 
			current.push(prt);
	}

	inline static double get_density_at(node* begin, vecnode& reserved_rad_nodes, particle* rsv_part = nullptr) {
		particle source = (rsv_part) ? *rsv_part : begin->mass_center;
		radius_node_catcher(begin, source.radius, &reserved_rad_nodes, (rsv_part) ? &rsv_part->position : nullptr);
		double sum = 0;
		for (auto& cur_node : reserved_rad_nodes) {
			auto pos_difference = source.position - cur_node->mass_center.position;
			auto max_radius = max(source.radius, cur_node->mass_center.radius);
			if (is_beyond_radius(pos_difference, max_radius))
				continue;
			sum += cur_node->mass_center.mass * grav_eq_utils::pressure_core(pos_difference, max_radius);
		}
		return sum;
	}

	inline static double get_energy_at(node* begin, vecnode& reserved_rad_nodes, vecnode& reserved_drn, particle* rsv_part = nullptr) {
		particle source = (rsv_part) ? *rsv_part : begin->mass_center;
		radius_node_catcher(begin, source.radius, &reserved_rad_nodes, (rsv_part) ? &rsv_part->position : nullptr);
		double sum = 0;
		for (auto& cur_node : reserved_rad_nodes) {
			auto pos_difference = source.position - cur_node->mass_center.position;
			auto max_radius = max(source.radius, cur_node->mass_center.radius);
			if (is_beyond_radius(pos_difference, max_radius))
				continue;
			sum += 
				(cur_node->mass_center.mass / get_density_at(cur_node, reserved_drn))
				* cur_node->mass_center.energy * grav_eq_utils::pressure_core(pos_difference, max_radius);
		}
		return sum;
	}

	inline static point grav_force(const particle& center, const particle& distant_prt) {
		constexpr double grav_const = 0.001;//just because ...
		auto t = grav_const * center.mass * distant_prt.mass * (distant_prt.position - center.position) /
			std::pow((distant_prt.radius + (center.position - distant_prt.position).norma2()), 1.5);
		//cout << t << endl;
		return t;
	}

	inline static point barnes_hutt_force_in_subtree(node* cur_node, const particle& current, const double error_edge_squared) {
		point gravitational_force = { 0,0 };
		std::stack<node*> cur_nodes;
		auto get_squared_error = [](const particle& cur, node* check_node) {
			return 0.5 * (check_node->leftbottom_corner - check_node->righttop_corner).norma2() / (cur.position - check_node->mass_center.position).norma2();
		};
		node** ptemp;
		while (true) {
			if (cur_node) {
				if (cur_node->particles_count_in_subtrees && get_squared_error(current,cur_node)>=error_edge_squared) {
					for (node::positioning i = node::positioning::leftbottom; i < node::positioning::null; ((int&)i)++) {
						if (*(ptemp = cur_node->get_dptr(i))) {
							cur_nodes.push(*ptemp);
						}
					}
				}
				else {
					gravitational_force +=
						grav_force(current, cur_node->mass_center);
				}
			}
			if (cur_nodes.size()) {
				cur_node = cur_nodes.top();
				cur_nodes.pop();
			}
			else
				break;
		}
		return gravitational_force;
	}

	inline static double get_pressure(double density, double energy, double polytropic_coef, double heat_capacity) {
		constexpr double big_C_coef = 8.3;
		return (heat_capacity - 1) * density * energy + big_C_coef*(polytropic_coef / 3. + 1. - heat_capacity) * std::pow(std::abs(density), polytropic_coef / 3. + 1.);
	}
	/*
	inline static point barnes_hutt_gravity(node* cur_node) {
		constexpr double error_edge_squared = 0.01;
		const particle current = cur_node->mass_center;
		point gravitational_force = { 0,0 };
		auto get_squared_error = [](const particle& cur, node* check_node) {
			return 0.5 * (check_node->leftbottom_corner - check_node->righttop_corner).norma2() / (cur.position - check_node->mass_center.position).norma2();
		};
		cone_node_extended_iterator iter(cur_node); 
		++iter;
		while (iter.is_iteratable()) {
			auto c_child = iter.get_cur_child();
			gravitational_force += 
				(get_squared_error(c_child->mass_center, c_child) < error_edge_squared) ?
				grav_force(current, c_child->mass_center) :
				barnes_hutt_force_in_subtree(c_child, c_child->mass_center, error_edge_squared);
			++iter;
		}
		return gravitational_force;
	}
	*/
	inline static bool is_beyond_radius(const point& dist, double radius) {
		return (dist.norma2() > radius* radius);
	}

	struct iteration_result {
		point dV;
		double dE;
		double dR;
		int interactions_count;
		double dT_CFL;
	};

	inline iteration_result iterate_particle(particle& current_prt, vecnode* rad_vector, vecnode* corad_vector1, vecnode* corad_vector2,
		const double heat_capacity, const double polytropic_coef, const double time_step) {
		constexpr double error_edge_squared = 0.05;
		constexpr double courant_number = 0.3;
		constexpr bool is_complete_SPH = true;
		node* cur_node = current.root_node; 
		int interactions_counter = 0;
		corad_vector1->clear();
		corad_vector2->clear();
		double cur_density = 0;
		double cur_energy = 0;
		double cur_pressure = 0;

		radius_node_catcher(cur_node, current_prt.radius, rad_vector, &current_prt.position);

		cur_density = get_density_at(cur_node, *corad_vector1, &current_prt);
		cur_energy = get_energy_at(cur_node, *corad_vector1, *corad_vector2, &current_prt);
		cur_pressure = get_pressure(cur_density, cur_energy, polytropic_coef, heat_capacity);

		point gravity = barnes_hutt_force_in_subtree(cur_node, current_prt, error_edge_squared);

		auto speed_of_sound = [&](double polytropic_coef, double energy) {
			constexpr double tuning_big_constant = 8.3;//8.3
			return std::sqrt(std::abs((1. + 1. / polytropic_coef) * tuning_big_constant * energy));
		};

		auto mu = [&](const particle& prt) {
			const point velocity_difference = (current_prt.velocity - prt.velocity);
			const point position_difference = (current_prt.position - prt.position);
			const double radius = max(current_prt.radius, prt.radius);
			double prod = velocity_difference * position_difference;
			double stabilizing_term = 0.01;
			if (prod < 0)
				return radius * prod / (
					(position_difference.norma2() + stabilizing_term)
				);
			else
				return 0.;
		};

		auto visc_tensor = [&](particle& prt, const double prt_energy, const double prt_density) {//tensor is not correct
			constexpr double alpha = 1;
			const point velocity_difference = (current_prt.velocity - prt.velocity);
			const point position_difference = (current_prt.position - prt.position);
			double omega = velocity_difference * position_difference;
			if (omega >= 0)
				return 0.;
			omega /= position_difference.norma();
			double cur_sos = speed_of_sound(polytropic_coef, cur_energy);
			double prt_sos = speed_of_sound(polytropic_coef, prt_energy);
			return -alpha * 0.5 * std::abs(cur_sos + prt_sos - 3 * omega) * omega / (cur_density + prt_density);
		};
		
		double dR = 0;
		double dE = 0;

		double max_mu = 0;
		double delta_time_CFL = 0;

		point dV = { 0,0 };
		double nabla_velocity = 0;

		dR = current_prt.radius * (0.05 + 0.45*(is_complete_SPH)) * (1. + std::pow(particle::desired_amount_of_interactions / (current_prt.interactions_count + 1), 0.33333));
		dR = max(dR, grav_eq_utils::epsilon * __size * 0.1);
		dR -= current_prt.radius;

		for (auto& it_node : *rad_vector) {
			auto pos_difference = current_prt.position - it_node->mass_center.position;
			auto vel_difference = current_prt.velocity - it_node->mass_center.velocity;
			auto max_radius = max(current_prt.radius, it_node->mass_center.radius);
			if (is_beyond_radius(pos_difference, max_radius) || pos_difference.norma2()<grav_eq_utils::epsilon || !is_complete_SPH)
				continue;
			auto inner_node_density = get_density_at(it_node, *corad_vector1);
			auto inner_node_energy = get_energy_at(it_node, *corad_vector1, *corad_vector2);
			auto inner_node_pressure = get_pressure(inner_node_density, inner_node_energy, polytropic_coef, heat_capacity);
			auto local_visc_tensor = 0*visc_tensor(it_node->mass_center, inner_node_energy, inner_node_density);
			auto core_gradient = grav_eq_utils::pressure_core_gradient(pos_difference, max_radius);

			//local_visc_tensor = ((local_visc_tensor > 0) ? 1 : -1)* std::sqrt(std::abs(local_visc_tensor));
			//printf("%lf\n", inner_node_pressure);

			max_mu = max(max_mu, mu(it_node->mass_center));
			nabla_velocity +=
				it_node->mass_center.mass * vel_difference * core_gradient;

			dV += it_node->mass_center.mass * (
				local_visc_tensor +
				inner_node_pressure / (inner_node_density * inner_node_density) +
				cur_pressure / (cur_density * cur_density)
				) * core_gradient;

			dE +=
				it_node->mass_center.mass * vel_difference * (
					local_visc_tensor +
					inner_node_pressure / (inner_node_density * inner_node_density) + 
					cur_pressure / (cur_density*cur_density)
				) * core_gradient;

			interactions_counter++;
		}

		nabla_velocity = -nabla_velocity / cur_density;
		delta_time_CFL = courant_number * current_prt.radius /
			(current_prt.radius * std::abs(nabla_velocity) + cur_energy + 1.2*(cur_energy + 0.5*max_mu));
		//dE *= (polytropic_coef - 1) / std::pow(std::abs(cur_density), polytropic_coef - 1) * (cur_density > 0 ? 1 : -1);

		return { -dV + gravity, (dE), dR, interactions_counter, delta_time_CFL };
	}

	inline particle iterate_over_particle(particle& current_prt, vecnode* rad_vector, vecnode* corad_vector1, vecnode* corad_vector2,
		const double heat_capacity, const double polytropic_coef, const double time_step) {// kind-of velvet integration
		
		particle local_prt = current_prt;
		double cfl_time = local_prt.cfl_time;
		double time_elapsed = 0;
		double local_time_step = time_step;
#define is_variable_timestep
#ifdef is_variable_timestep
		double min_cfl = INFINITY;
		local_time_step = time_step/(std::pow(2,std::log2(std::ceil(time_step/cfl_time))));
		do {
#endif
			//if(local_time_step != time_step)
				//std::cout << local_time_step << endl;
			auto ans = iterate_particle(local_prt, rad_vector, corad_vector1, corad_vector2, heat_capacity, polytropic_coef, local_time_step);
			local_prt.energy += local_time_step * ans.dE;
			local_prt.interactions_count = ans.interactions_count;
			local_prt.radius += 0.5 * ans.dR;

			point initial_vel = local_prt.velocity;
			local_prt.position += local_time_step * (local_prt.velocity + local_time_step * (
				(2. / 3) * ans.dV - (1. / 6) * local_prt.acceleration
				));
			local_prt.velocity += local_time_step * (1.5 * ans.dV - 0.5 * local_prt.acceleration);

			auto n_ans = iterate_particle(local_prt, rad_vector, corad_vector1, corad_vector2, polytropic_coef, heat_capacity, local_time_step);
			//local_prt.energy += 0.25 * time_step * n_ans.dE;
			local_prt.interactions_count = n_ans.interactions_count;
			local_prt.radius = //min(
				max(local_prt.radius + 0.5 * n_ans.dR, grav_eq_utils::epsilon)
				//,50
				;//);
			local_prt.energy = 1 * local_time_step * (ans.dE);

			local_prt.acceleration = (ans.dV + n_ans.dV) * 0.5;
			local_prt.velocity = initial_vel + local_time_step * ((1. / 3) * n_ans.dV + (5. / 6) * ans.dV - (1. / 6) * local_prt.acceleration);
			time_elapsed += local_time_step;

#ifdef is_variable_timestep
			min_cfl = min(ans.dT_CFL, min_cfl);
		} while (time_elapsed<min(cfl_time, local_time_step));
#endif
		local_prt.cfl_time = cfl_time;
		return local_prt;
	}

	inline void iterate_subtree(node* subtree_root, std::stack<node*>* cur_nodes, vecnode* rad_nodes, vecnode* first_corad, vecnode* second_corad) {
		while (cur_nodes->size())
			cur_nodes->pop();
		node* cur_node = subtree_root;
		node** ptemp;
		while (true) {
			if (cur_node) {
				if (cur_node->particles_count_in_subtrees) {
					for (node::positioning i = node::positioning::leftbottom; i < node::positioning::null; ((int&)i)++) {
						if (*(ptemp = cur_node->get_dptr(i))) {
							cur_nodes->push(*ptemp);
						}
					}
				}
				else {
					if (std::abs(cur_node->mass_center.mass) > grav_eq_utils::epsilon && !cur_node->particles_count_in_subtrees) {
						auto prt = iterate_over_particle(cur_node->mass_center, rad_nodes, first_corad, second_corad, heat_capacity, polytropic_coef, time_step);
						cur_node->mass_center.visited = flickering;
						if (prt.velocity[0] == prt.velocity[0] && prt.acceleration[0] == prt.acceleration[0]) {
							buffer_mutex.lock();
							buffer.push(prt); 
							buffer_mutex.unlock();
						}
						else
							printf("nan\n");
					}
					//current.root_node = t;
					//printf("asd");
				}
			}
			if (cur_nodes->size()) {
				cur_node->mass_center.visited = flickering;
				cur_node = cur_nodes->top();
				cur_nodes->pop();
			}
			else
				break;
		}
	}

	inline void start_single_thread() {
		std::thread proc([&]() {
			vecnode rad_nodes, first_corad, second_corad;
			std::stack <node*> cur_nodes;
			while (true) {
				printf("%i particles\n", current.root_node->particles_count_in_subtrees);
				//Sleep(5000);
				iterate_subtree(current.root_node, &cur_nodes, &rad_nodes, &first_corad, &second_corad);
				//Sleep(50000000);
				pause.lock();
				pause.unlock();
				pre_swap.lock();
				current.clear();
				current.swap(buffer);
 				pre_swap.unlock();
				//Sleep(5000);
			}
		});
		proc.detach();
	}

	inline void subdivide_tree() {
		constexpr int catch_level = 5;
		const double relation = (double)current.root_node->particles_count_in_subtrees / num_of_threads;

		int cur_thread_num = 0;
		int cur_am_of_particles = 0;
		pair<node*, int> cur_node = { current.root_node , 0 };
		node* temp = nullptr;

		_subdivision_roots.clear();
		while (_subdivision_cur_nodes.size()) 
			_subdivision_cur_nodes.pop();

		printf("%i particles\n", current.root_node->particles_count_in_subtrees);

		while (true) {
			if (cur_node.first) {
				if (cur_node.second < catch_level && cur_node.first->particles_count_in_subtrees) {
					for (quad_tree::positioning i = quad_tree::positioning::leftbottom; i < quad_tree::positioning::null; ((int&)i)++) {
						if ((temp = cur_node.first->get(i))) {
							_subdivision_cur_nodes.push({ temp, cur_node.second + 1 });
						}
					}
				}
				else {
					cur_am_of_particles += (!cur_node.first->particles_count_in_subtrees) + cur_node.first->particles_count_in_subtrees;
					if (cur_am_of_particles / relation - 1 > cur_thread_num)
						cur_thread_num++;
					_subdivision_roots.push_back({ cur_thread_num, cur_node.first });
				}
			}
			if (_subdivision_cur_nodes.size()) {
				cur_node = _subdivision_cur_nodes.top();
				_subdivision_cur_nodes.pop();
			}
			else
				break;
		}

		for (auto& vec : threads_desired_roots)
			vec.clear();

		for (auto& root : _subdivision_roots)
			threads_desired_roots[root.first].push_back(root.second);
	}
	
	inline void start_threads() {
		if (threads.size())
			return;

		subdivide_tree();
		for (int i = 0; i < num_of_threads; i++){
			threads.push_back(new pooled_thread()); // executors
			auto t = threads.back()->__void_ptr_accsess();
			*t = (void*)i;
			threads.back()->set_new_function([this](void** void_ptr) {

				typedef struct {
					vecnode rad_nodes, first_corad, second_corad;
					std::stack <node*> cur_nodes;
					vecnode* root_ptrs;
					int id;
				} thread_info;
				thread_info** pptr = (thread_info**)void_ptr;

				if (*pptr < (thread_info*)0x400) {
					int id = (int)*pptr;
					*pptr = new thread_info;
					(*pptr)->root_ptrs = &(threads_desired_roots[id]);
					(*pptr)->id = id;
				}

				pause.lock();
				pause.unlock();

				for (auto& local_root : *(*pptr)->root_ptrs)
					iterate_subtree(local_root, &(*pptr)->cur_nodes, &(*pptr)->rad_nodes, &(*pptr)->first_corad, &(*pptr)->second_corad);

				//printf("thread finished\n");

			});
			threads.back()->sign_awaiting();
		}

		Sleep(5);

		threads.push_back(new pooled_thread());//observer
		threads.back()->set_new_awaiting_time(10);
		threads.back()->set_new_default_state(pooled_thread::state::waiting);
		threads.back()->set_new_function([this](void** ptr) {
			for (auto ptr : threads)
				if (ptr != threads.back() && ptr->get_state() != pooled_thread::state::idle) {
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
		threads.back()->sign_awaiting();
	}
};
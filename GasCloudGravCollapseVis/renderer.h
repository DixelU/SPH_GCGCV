#pragma once

#include "simulation.h"

#include <cstddef>
#include <vector>

namespace app
{

struct Matrix4
{
	float values[16]{};
};

struct Camera
{
	sph::Vec3 target{0.f, 0.f, 0.f};
	float yaw = 0.65f;
	float pitch = 0.35f;
	float distance = 125.f;
	float field_of_view_degrees = 45.f;

	[[nodiscard]] sph::Vec3 position() const;
	[[nodiscard]] Matrix4 view_matrix() const;
	[[nodiscard]] Matrix4 projection_matrix(float aspect_ratio) const;
	void orbit(float delta_x, float delta_y);
	void pan(float delta_x, float delta_y, float viewport_height);
	void zoom(float wheel_delta);
	void reset(float domain_size);
};

struct VisualizationSettings
{
	sph::Field field = sph::Field::density;
	float brightness = 1.f;
	float point_size = 4.f;
	float background[3]{0.012f, 0.016f, 0.028f};
	bool logarithmic_scale = true;
	bool invert_color_map = false;
	bool show_domain_box = true;
};

class ParticleRenderer
{
public:
	ParticleRenderer();
	~ParticleRenderer();

	ParticleRenderer(const ParticleRenderer&) = delete;
	ParticleRenderer& operator=(const ParticleRenderer&) = delete;

	void draw(
		const sph::SimulationSnapshot& snapshot,
		const Camera& camera,
		const VisualizationSettings& settings,
		int framebuffer_width,
		int framebuffer_height,
		float domain_size);

private:
	struct Vertex
	{
		float position[3];
		float color[3];
	};

	void upload_particles(
		const sph::SimulationSnapshot& snapshot,
		const VisualizationSettings& settings);
	void create_domain_box(float domain_size);

	unsigned int program_ = 0;
	unsigned int particle_vertex_array_ = 0;
	unsigned int particle_buffer_ = 0;
	unsigned int box_vertex_array_ = 0;
	unsigned int box_buffer_ = 0;
	int mvp_location_ = -1;
	int point_size_location_ = -1;
	int round_points_location_ = -1;
	std::size_t particle_vertex_count_ = 0;
	std::size_t box_vertex_count_ = 0;
	float box_domain_size_ = -1.f;
	std::vector<Vertex> staging_;
};

} // namespace app

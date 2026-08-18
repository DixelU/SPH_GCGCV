#include "pch.h"

#include <GLFW/glfw3.h>
#include <GL/glext.h>

#include "renderer.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace app
{
namespace
{

struct OpenGlApi
{
	PFNGLCREATESHADERPROC create_shader = nullptr;
	PFNGLSHADERSOURCEPROC shader_source = nullptr;
	PFNGLCOMPILESHADERPROC compile_shader = nullptr;
	PFNGLGETSHADERIVPROC get_shader_iv = nullptr;
	PFNGLGETSHADERINFOLOGPROC get_shader_info_log = nullptr;
	PFNGLDELETESHADERPROC delete_shader = nullptr;
	PFNGLCREATEPROGRAMPROC create_program = nullptr;
	PFNGLATTACHSHADERPROC attach_shader = nullptr;
	PFNGLLINKPROGRAMPROC link_program = nullptr;
	PFNGLGETPROGRAMIVPROC get_program_iv = nullptr;
	PFNGLGETPROGRAMINFOLOGPROC get_program_info_log = nullptr;
	PFNGLDELETEPROGRAMPROC delete_program = nullptr;
	PFNGLGETUNIFORMLOCATIONPROC get_uniform_location = nullptr;
	PFNGLGENVERTEXARRAYSPROC gen_vertex_arrays = nullptr;
	PFNGLGENBUFFERSPROC gen_buffers = nullptr;
	PFNGLBINDVERTEXARRAYPROC bind_vertex_array = nullptr;
	PFNGLBINDBUFFERPROC bind_buffer = nullptr;
	PFNGLENABLEVERTEXATTRIBARRAYPROC enable_vertex_attrib_array = nullptr;
	PFNGLVERTEXATTRIBPOINTERPROC vertex_attrib_pointer = nullptr;
	PFNGLDELETEBUFFERSPROC delete_buffers = nullptr;
	PFNGLDELETEVERTEXARRAYSPROC delete_vertex_arrays = nullptr;
	PFNGLBUFFERDATAPROC buffer_data = nullptr;
	PFNGLUSEPROGRAMPROC use_program = nullptr;
	PFNGLUNIFORMMATRIX4FVPROC uniform_matrix4fv = nullptr;
	PFNGLUNIFORM1IPROC uniform1i = nullptr;

	[[nodiscard]] bool load()
	{
		bool success = true;
		auto load_one = [&](auto& function, const char* name)
		{
			using function_type = std::remove_reference_t<decltype(function)>;
			function = reinterpret_cast<function_type>(glfwGetProcAddress(name));
			success = success && function != nullptr;
		};
		load_one(create_shader, "glCreateShader");
		load_one(shader_source, "glShaderSource");
		load_one(compile_shader, "glCompileShader");
		load_one(get_shader_iv, "glGetShaderiv");
		load_one(get_shader_info_log, "glGetShaderInfoLog");
		load_one(delete_shader, "glDeleteShader");
		load_one(create_program, "glCreateProgram");
		load_one(attach_shader, "glAttachShader");
		load_one(link_program, "glLinkProgram");
		load_one(get_program_iv, "glGetProgramiv");
		load_one(get_program_info_log, "glGetProgramInfoLog");
		load_one(delete_program, "glDeleteProgram");
		load_one(get_uniform_location, "glGetUniformLocation");
		load_one(gen_vertex_arrays, "glGenVertexArrays");
		load_one(gen_buffers, "glGenBuffers");
		load_one(bind_vertex_array, "glBindVertexArray");
		load_one(bind_buffer, "glBindBuffer");
		load_one(enable_vertex_attrib_array, "glEnableVertexAttribArray");
		load_one(vertex_attrib_pointer, "glVertexAttribPointer");
		load_one(delete_buffers, "glDeleteBuffers");
		load_one(delete_vertex_arrays, "glDeleteVertexArrays");
		load_one(buffer_data, "glBufferData");
		load_one(use_program, "glUseProgram");
		load_one(uniform_matrix4fv, "glUniformMatrix4fv");
		load_one(uniform1i, "glUniform1i");
		return success;
	}
};

OpenGlApi gl_api;

#define glCreateShader gl_api.create_shader
#define glShaderSource gl_api.shader_source
#define glCompileShader gl_api.compile_shader
#define glGetShaderiv gl_api.get_shader_iv
#define glGetShaderInfoLog gl_api.get_shader_info_log
#define glDeleteShader gl_api.delete_shader
#define glCreateProgram gl_api.create_program
#define glAttachShader gl_api.attach_shader
#define glLinkProgram gl_api.link_program
#define glGetProgramiv gl_api.get_program_iv
#define glGetProgramInfoLog gl_api.get_program_info_log
#define glDeleteProgram gl_api.delete_program
#define glGetUniformLocation gl_api.get_uniform_location
#define glGenVertexArrays gl_api.gen_vertex_arrays
#define glGenBuffers gl_api.gen_buffers
#define glBindVertexArray gl_api.bind_vertex_array
#define glBindBuffer gl_api.bind_buffer
#define glEnableVertexAttribArray gl_api.enable_vertex_attrib_array
#define glVertexAttribPointer gl_api.vertex_attrib_pointer
#define glDeleteBuffers gl_api.delete_buffers
#define glDeleteVertexArrays gl_api.delete_vertex_arrays
#define glBufferData gl_api.buffer_data
#define glUseProgram gl_api.use_program
#define glUniformMatrix4fv gl_api.uniform_matrix4fv
#define glUniform1i gl_api.uniform1i

[[nodiscard]] sph::Vec3 cross(const sph::Vec3& lhs, const sph::Vec3& rhs)
{
	return {
		lhs[1] * rhs[2] - lhs[2] * rhs[1],
		lhs[2] * rhs[0] - lhs[0] * rhs[2],
		lhs[0] * rhs[1] - lhs[1] * rhs[0]};
}

[[nodiscard]] sph::Vec3 normalized(const sph::Vec3& value)
{
	const float length = value.get_norm();
	return length > 0.f ? value / length : sph::Vec3{};
}

[[nodiscard]] Matrix4 multiply(const Matrix4& lhs, const Matrix4& rhs)
{
	Matrix4 result;
	for (int column = 0; column < 4; ++column)
		for (int row = 0; row < 4; ++row)
			for (int inner = 0; inner < 4; ++inner)
				result.values[column * 4 + row] +=
					lhs.values[inner * 4 + row] * rhs.values[column * 4 + inner];
	return result;
}

[[nodiscard]] GLuint compile_shader(GLenum type, const char* source)
{
	const GLuint shader = glCreateShader(type);
	glShaderSource(shader, 1, &source, nullptr);
	glCompileShader(shader);
	GLint compiled = GL_FALSE;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
	if (compiled == GL_TRUE)
		return shader;
	GLint length = 0;
	glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length);
	std::string log(static_cast<std::size_t>(std::max(length, 1)), '\0');
	glGetShaderInfoLog(shader, length, nullptr, log.data());
	glDeleteShader(shader);
	throw std::runtime_error("OpenGL shader compilation failed: " + log);
}

[[nodiscard]] GLuint create_program()
{
	static constexpr const char* vertex_source = R"glsl(
#version 330 core
layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_color;
uniform mat4 u_mvp;
uniform int u_point_size;
out vec3 vertex_color;
void main()
{
    gl_Position = u_mvp * vec4(in_position, 1.0);
    gl_PointSize = float(u_point_size);
    vertex_color = in_color;
}
)glsl";
	static constexpr const char* fragment_source = R"glsl(
#version 330 core
in vec3 vertex_color;
uniform int u_round_points;
out vec4 out_color;
void main()
{
    float alpha = 1.0;
    if (u_round_points != 0)
    {
        vec2 centered = gl_PointCoord * 2.0 - 1.0;
        float radius2 = dot(centered, centered);
        if (radius2 > 1.0)
            discard;
        alpha = 1.0 - smoothstep(0.55, 1.0, radius2);
    }
    out_color = vec4(vertex_color, alpha);
}
)glsl";
	const GLuint vertex = compile_shader(GL_VERTEX_SHADER, vertex_source);
	const GLuint fragment = compile_shader(GL_FRAGMENT_SHADER, fragment_source);
	const GLuint program = glCreateProgram();
	glAttachShader(program, vertex);
	glAttachShader(program, fragment);
	glLinkProgram(program);
	glDeleteShader(vertex);
	glDeleteShader(fragment);
	GLint linked = GL_FALSE;
	glGetProgramiv(program, GL_LINK_STATUS, &linked);
	if (linked == GL_TRUE)
		return program;
	GLint length = 0;
	glGetProgramiv(program, GL_INFO_LOG_LENGTH, &length);
	std::string log(static_cast<std::size_t>(std::max(length, 1)), '\0');
	glGetProgramInfoLog(program, length, nullptr, log.data());
	glDeleteProgram(program);
	throw std::runtime_error("OpenGL program link failed: " + log);
}

[[nodiscard]] std::array<float, 3> sequential_color(float value)
{
	const float t = std::clamp(value, 0.f, 1.f);
	const std::array<float, 3> deep{0.035f, 0.08f, 0.32f};
	const std::array<float, 3> cyan{0.02f, 0.72f, 0.82f};
	const std::array<float, 3> yellow{1.f, 0.82f, 0.12f};
	const std::array<float, 3> white{1.f, 0.98f, 0.92f};
	auto blend = [](const auto& a, const auto& b, float amount)
	{
		return std::array<float, 3>{
			a[0] + (b[0] - a[0]) * amount,
			a[1] + (b[1] - a[1]) * amount,
			a[2] + (b[2] - a[2]) * amount};
	};
	if (t < 0.4f)
		return blend(deep, cyan, t / 0.4f);
	if (t < 0.8f)
		return blend(cyan, yellow, (t - 0.4f) / 0.4f);
	return blend(yellow, white, (t - 0.8f) / 0.2f);
}

[[nodiscard]] std::array<float, 3> diverging_color(float value)
{
	const float t = std::clamp(value, -1.f, 1.f);
	if (t < 0.f)
		return {0.18f + 0.72f * (1.f + t), 0.35f + 0.55f * (1.f + t), 1.f};
	return {1.f, 0.90f - 0.70f * t, 0.90f - 0.78f * t};
}

} // namespace

sph::Vec3 Camera::position() const
{
	const float cosine_pitch = std::cos(pitch);
	return target + distance * sph::Vec3{
		cosine_pitch * std::sin(yaw),
		std::sin(pitch),
		cosine_pitch * std::cos(yaw)};
}

Matrix4 Camera::view_matrix() const
{
	const sph::Vec3 eye = position();
	const sph::Vec3 forward = normalized(target - eye);
	const sph::Vec3 side = normalized(cross(forward, {0.f, 1.f, 0.f}));
	const sph::Vec3 up = cross(side, forward);
	Matrix4 result;
	result.values[0] = side[0];
	result.values[4] = side[1];
	result.values[8] = side[2];
	result.values[1] = up[0];
	result.values[5] = up[1];
	result.values[9] = up[2];
	result.values[2] = -forward[0];
	result.values[6] = -forward[1];
	result.values[10] = -forward[2];
	result.values[12] = -(side * eye);
	result.values[13] = -(up * eye);
	result.values[14] = forward * eye;
	result.values[15] = 1.f;
	return result;
}

Matrix4 Camera::projection_matrix(float aspect_ratio) const
{
	const float near_plane = 0.05f;
	const float far_plane = std::max(10000.f, distance * 20.f);
	const float radians = field_of_view_degrees * sph::pi / 180.f;
	const float inverse_tangent = 1.f / std::tan(radians * 0.5f);
	Matrix4 result;
	result.values[0] = inverse_tangent / std::max(aspect_ratio, 0.01f);
	result.values[5] = inverse_tangent;
	result.values[10] = (far_plane + near_plane) / (near_plane - far_plane);
	result.values[11] = -1.f;
	result.values[14] = (2.f * far_plane * near_plane) / (near_plane - far_plane);
	return result;
}

void Camera::orbit(float delta_x, float delta_y)
{
	yaw -= delta_x * 0.006f;
	pitch = std::clamp(pitch - delta_y * 0.006f, -1.52f, 1.52f);
}

void Camera::pan(float delta_x, float delta_y, float viewport_height)
{
	const sph::Vec3 eye = position();
	const sph::Vec3 forward = normalized(target - eye);
	const sph::Vec3 right = normalized(cross(forward, {0.f, 1.f, 0.f}));
	const sph::Vec3 up = normalized(cross(right, forward));
	const float world_per_pixel = 2.f * distance *
		std::tan(field_of_view_degrees * sph::pi / 360.f) /
		std::max(viewport_height, 1.f);
	target += (-delta_x * right + delta_y * up) * world_per_pixel;
}

void Camera::zoom(float wheel_delta)
{
	distance = std::clamp(distance * std::exp(-wheel_delta * 0.14f), 0.05f, 100000.f);
}

void Camera::reset(float domain_size)
{
	target = {0.f, 0.f, 0.f};
	yaw = 0.65f;
	pitch = 0.35f;
	distance = domain_size * 1.25f;
}

ParticleRenderer::ParticleRenderer()
{
	if (!gl_api.load())
		throw std::runtime_error("Could not initialize the OpenGL function loader");
	program_ = create_program();
	mvp_location_ = glGetUniformLocation(program_, "u_mvp");
	point_size_location_ = glGetUniformLocation(program_, "u_point_size");
	round_points_location_ = glGetUniformLocation(program_, "u_round_points");

	glGenVertexArrays(1, &particle_vertex_array_);
	glGenBuffers(1, &particle_buffer_);
	glBindVertexArray(particle_vertex_array_);
	glBindBuffer(GL_ARRAY_BUFFER, particle_buffer_);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void*>(offsetof(Vertex, position)));
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void*>(offsetof(Vertex, color)));

	glGenVertexArrays(1, &box_vertex_array_);
	glGenBuffers(1, &box_buffer_);
	glBindVertexArray(box_vertex_array_);
	glBindBuffer(GL_ARRAY_BUFFER, box_buffer_);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void*>(offsetof(Vertex, position)));
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void*>(offsetof(Vertex, color)));
	glBindVertexArray(0);
}

ParticleRenderer::~ParticleRenderer()
{
	if (box_buffer_)
		glDeleteBuffers(1, &box_buffer_);
	if (box_vertex_array_)
		glDeleteVertexArrays(1, &box_vertex_array_);
	if (particle_buffer_)
		glDeleteBuffers(1, &particle_buffer_);
	if (particle_vertex_array_)
		glDeleteVertexArrays(1, &particle_vertex_array_);
	if (program_)
		glDeleteProgram(program_);
}

void ParticleRenderer::upload_particles(
	const sph::SimulationSnapshot& snapshot,
	const VisualizationSettings& settings)
{
	staging_.resize(snapshot.particles.size());
	float maximum_energy = 0.f;
	for (const sph::RenderParticle& particle : snapshot.particles)
		maximum_energy = std::max(maximum_energy, std::abs(particle.energy));
	const float maximum_for_field = [&]
	{
		switch (settings.field)
		{
			case sph::Field::density: return snapshot.maximum_density;
			case sph::Field::energy: return maximum_energy;
			case sph::Field::speed: return snapshot.maximum_speed;
			case sph::Field::acceleration: return snapshot.maximum_acceleration;
			case sph::Field::x_velocity:
			case sph::Field::y_velocity:
			case sph::Field::z_velocity: return snapshot.maximum_speed;
		}
		return 1.f;
	}();
	const float safe_maximum = std::max(maximum_for_field, 1e-12f);
	for (std::size_t index = 0; index < snapshot.particles.size(); ++index)
	{
		const sph::RenderParticle& particle = snapshot.particles[index];
		float value = 0.f;
		bool signed_field = false;
		switch (settings.field)
		{
			case sph::Field::density: value = particle.density; break;
			case sph::Field::energy: value = particle.energy; break;
			case sph::Field::speed: value = particle.speed; break;
			case sph::Field::acceleration: value = particle.acceleration; break;
			case sph::Field::x_velocity: value = particle.velocity[0]; signed_field = true; break;
			case sph::Field::y_velocity: value = particle.velocity[1]; signed_field = true; break;
			case sph::Field::z_velocity: value = particle.velocity[2]; signed_field = true; break;
		}
		float normalized = value / safe_maximum;
		if (settings.logarithmic_scale && !signed_field)
			normalized = std::log1p(std::max(normalized, 0.f) * 30.f) / std::log(31.f);
		normalized *= settings.brightness;
		const auto color = signed_field ? diverging_color(normalized) : sequential_color(normalized);
		staging_[index] = {
			{particle.position[0], particle.position[1], particle.position[2]},
			{color[0], color[1], color[2]}};
	}
	glBindBuffer(GL_ARRAY_BUFFER, particle_buffer_);
	glBufferData(
		GL_ARRAY_BUFFER,
		static_cast<GLsizeiptr>(staging_.size() * sizeof(Vertex)),
		staging_.data(),
		GL_STREAM_DRAW);
	particle_vertex_count_ = staging_.size();
}

void ParticleRenderer::create_domain_box(float domain_size)
{
	if (box_domain_size_ == domain_size)
		return;
	box_domain_size_ = domain_size;
	const float h = domain_size * 0.5f;
	const std::array<sph::Vec3, 8> corners{
		sph::Vec3{-h, -h, -h}, sph::Vec3{ h, -h, -h},
		sph::Vec3{-h,  h, -h}, sph::Vec3{ h,  h, -h},
		sph::Vec3{-h, -h,  h}, sph::Vec3{ h, -h,  h},
		sph::Vec3{-h,  h,  h}, sph::Vec3{ h,  h,  h}};
	static constexpr std::array<std::array<int, 2>, 12> edges{{
		{0,1}, {0,2}, {0,4}, {1,3}, {1,5}, {2,3},
		{2,6}, {3,7}, {4,5}, {4,6}, {5,7}, {6,7}}};
	std::vector<Vertex> vertices;
	vertices.reserve(edges.size() * 2);
	for (const auto& edge : edges)
		for (int corner : edge)
			vertices.push_back({
				{corners[corner][0], corners[corner][1], corners[corner][2]},
				{0.20f, 0.28f, 0.40f}});
	glBindBuffer(GL_ARRAY_BUFFER, box_buffer_);
	glBufferData(
		GL_ARRAY_BUFFER,
		static_cast<GLsizeiptr>(vertices.size() * sizeof(Vertex)),
		vertices.data(),
		GL_STREAM_DRAW);
	box_vertex_count_ = vertices.size();
}

void ParticleRenderer::draw(
	const sph::SimulationSnapshot& snapshot,
	const Camera& camera,
	const VisualizationSettings& settings,
	int framebuffer_width,
	int framebuffer_height,
	float domain_size)
{
	if (framebuffer_width <= 0 || framebuffer_height <= 0)
		return;
	upload_particles(snapshot, settings);
	create_domain_box(domain_size);
	const Matrix4 mvp = multiply(
		camera.projection_matrix(
			static_cast<float>(framebuffer_width) / framebuffer_height),
		camera.view_matrix());

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_PROGRAM_POINT_SIZE);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glUseProgram(program_);
	glUniformMatrix4fv(mvp_location_, 1, GL_FALSE, mvp.values);
	glUniform1i(point_size_location_, static_cast<int>(std::round(settings.point_size)));
	glUniform1i(round_points_location_, 1);
	glBindVertexArray(particle_vertex_array_);
	glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(particle_vertex_count_));

	if (settings.show_domain_box)
	{
		glUniform1i(round_points_location_, 0);
		glBindVertexArray(box_vertex_array_);
		glDrawArrays(GL_LINES, 0, static_cast<GLsizei>(box_vertex_count_));
	}
	glBindVertexArray(0);
	glUseProgram(0);
}

} // namespace app

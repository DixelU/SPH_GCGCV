#include "pch.h"

#include <GLFW/glfw3.h>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include "renderer.h"
#include "simulation.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <string>
#include <vector>

namespace
{

[[nodiscard]] bool nearly_equal(float lhs, float rhs, float tolerance = 1e-5f)
{
	return std::abs(lhs - rhs) <= tolerance;
}

int run_numerical_self_tests()
{
	int failures = 0;
	auto check = [&](bool condition, const char* name)
	{
		std::printf("[%s] %s\n", condition ? "PASS" : "FAIL", name);
		if (!condition)
			++failures;
	};

	const float smoothing_length = 1.3f;
	const float sample_radius = 0.37f;
	const float difference_step = 1e-4f;
	const sph::Vec3 sample{sample_radius, 0.f, 0.f};
	const sph::Vec3 gradient = sph::wendland_c2_gradient(sample, smoothing_length);
	const float numerical_gradient =
		(sph::wendland_c2(sample_radius + difference_step, smoothing_length) -
		 sph::wendland_c2(sample_radius - difference_step, smoothing_length)) /
		(2.f * difference_step);
	check(
		nearly_equal(gradient[0], numerical_gradient, 2e-3f) &&
		nearly_equal(gradient[1], 0.f) && nearly_equal(gradient[2], 0.f) &&
		sph::wendland_c2_gradient({0.f, 0.f, 0.f}, 1.f).get_norm2() == 0.f,
		"3D Wendland gradient matches finite differences and is smooth at the origin");

	constexpr int integration_steps = 16384;
	const float dr = smoothing_length / integration_steps;
	float integral = 0.f;
	for (int step = 0; step < integration_steps; ++step)
	{
		const float radius = (step + 0.5f) * dr;
		integral += 4.f * sph::pi * radius * radius *
			sph::wendland_c2(radius, smoothing_length) * dr;
	}
	check(
		nearly_equal(integral, 1.f, 3e-4f) &&
		sph::wendland_c2(smoothing_length, smoothing_length) == 0.f,
		"3D Wendland kernel is normalized over volume and compactly supported");

	sph::InitialConditions cloud_settings;
	cloud_settings.particle_count = 1000;
	cloud_settings.seed = 42;
	const auto cloud = sph::make_rotating_cloud(cloud_settings);
	bool cloud_is_valid = cloud.size() == cloud_settings.particle_count;
	const float cloud_radius = cloud_settings.domain_size * cloud_settings.cloud_radius_fraction;
	for (const sph::Particle& particle : cloud)
		cloud_is_valid = cloud_is_valid && particle.position.get_norm() <= cloud_radius &&
			nearly_equal(particle.velocity[2], 0.f);
	check(cloud_is_valid, "spherical initializer produces the requested particle count and axial rotation");

	std::vector<sph::Particle> lattice;
	for (int z = 0; z < 4; ++z)
		for (int y = 0; y < 4; ++y)
			for (int x = 0; x < 4; ++x)
				lattice.push_back(sph::Particle{
					.position = {-3.f + 2.f * x, -3.f + 2.f * y, -3.f + 2.f * z},
					.velocity = {},
					.acceleration = {},
					.mass = 1.f,
					.smoothing_length = 2.6f,
					.energy = 1.f,
					.cfl_time = 1.f,
					.interaction_count = 1});
	sph::SimulationConfig no_forces;
	no_forces.enable_gravity = false;
	no_forces.enable_hydrodynamics = false;
	sph::Simulation3D lattice_simulation(lattice, 16.f, no_forces);
	lattice_simulation.prepare();
	bool densities_match = true;
	for (std::size_t first = 0; first < lattice.size(); ++first)
	{
		float brute_density = 0.f;
		for (const sph::Particle& second : lattice)
		{
			const float support = std::max(
				lattice[first].smoothing_length,
				second.smoothing_length);
			brute_density += second.mass * sph::wendland_c2(
				lattice[first].position - second.position,
				support);
		}
		densities_match = densities_match && nearly_equal(
			lattice_simulation.densities()[first],
			brute_density,
			1e-4f * std::max(1.f, brute_density));
	}
	check(densities_match, "3D spatial hash density matches brute force");

	sph::Particle light{
		.position = {-1.f, 0.f, 0.5f}, .mass = 2.f, .smoothing_length = 0.5f};
	sph::Particle heavy = light;
	heavy.mass = 2000.f;
	sph::Particle distant{
		.position = {1.f, 0.f, -0.5f}, .mass = 5.f, .smoothing_length = 0.5f};
	const sph::Vec3 light_acceleration = sph::Simulation3D::pair_gravity(light, distant, 0.001f);
	const sph::Vec3 heavy_acceleration = sph::Simulation3D::pair_gravity(heavy, distant, 0.001f);
	const sph::Vec3 reverse_acceleration = sph::Simulation3D::pair_gravity(distant, light, 0.001f);
	check(
		(light_acceleration - heavy_acceleration).get_norm() < 1e-7f,
		"gravity acceleration is independent of accelerated mass");
	check(
		(light.mass * light_acceleration + distant.mass * reverse_acceleration).get_norm() < 1e-6f,
		"3D pair gravity conserves linear momentum");

	sph::Particle inertial{
		.position = {1.f, 2.f, 3.f},
		.velocity = {3.f, -4.f, 5.f},
		.acceleration = {},
		.mass = 1.f,
		.smoothing_length = 1.f,
		.energy = 1.f,
		.cfl_time = 1.f};
	no_forces.maximum_time_step = 0.01f;
	sph::Simulation3D inertial_simulation({inertial}, 100.f, no_forces);
	(void)inertial_simulation.step();
	const sph::Particle& inertial_result = inertial_simulation.particles().front();
	check(
		nearly_equal(inertial_result.position[0], 1.03f) &&
		nearly_equal(inertial_result.position[1], 1.96f) &&
		nearly_equal(inertial_result.position[2], 3.05f) &&
		(inertial_result.velocity - inertial.velocity).get_norm2() == 0.f,
		"3D symplectic integrator drifts an inertial particle exactly once");

	sph::SimulationConfig gravity_only;
	gravity_only.enable_hydrodynamics = false;
	gravity_only.maximum_time_step = 0.001f;
	sph::Particle first{
		.position = {-2.f, 0.f, 0.25f},
		.velocity = {0.f, 0.02f, 0.f},
		.mass = 2.f,
		.smoothing_length = 0.2f,
		.energy = 1.f,
		.cfl_time = 1.f};
	sph::Particle second{
		.position = {2.f, 0.f, -0.25f},
		.velocity = {0.f, -0.008f, 0.f},
		.mass = 5.f,
		.smoothing_length = 0.2f,
		.energy = 1.f,
		.cfl_time = 1.f};
	const sph::Vec3 initial_momentum = first.mass * first.velocity + second.mass * second.velocity;
	sph::Simulation3D pair_simulation({first, second}, 100.f, gravity_only);
	bool finite_pair = true;
	for (int step = 0; step < 1000; ++step)
	{
		(void)pair_simulation.step();
		for (const sph::Particle& particle : pair_simulation.particles())
			for (std::size_t axis = 0; axis < 3; ++axis)
				finite_pair = finite_pair && std::isfinite(particle.position[axis]);
	}
	const auto pair_snapshot = pair_simulation.make_snapshot();
	check(
		finite_pair && (pair_snapshot.momentum - initial_momentum).get_norm() < 1e-4f,
		"long 3D two-body run remains finite and conserves momentum");

	std::printf("%s: %d failure(s)\n",
		failures ? "SELF-TEST FAILED" : "SELF-TEST PASSED",
		failures);
	return failures ? 1 : 0;
}

int run_headless(int steps, std::uint32_t seed, std::size_t particle_count)
{
	sph::InitialConditions initial;
	initial.particle_count = particle_count;
	initial.seed = seed;
	sph::Simulation3D simulation(
		sph::make_rotating_cloud(initial),
		initial.domain_size);
	const int report_interval = std::max(steps / 10, 1);
	for (int step = 0; step < steps; ++step)
	{
		const auto begin = std::chrono::steady_clock::now();
		(void)simulation.step();
		const float elapsed_ms = static_cast<float>(
			std::chrono::duration<double, std::milli>(
				std::chrono::steady_clock::now() - begin).count());
		if ((step + 1) % report_interval == 0 || step + 1 == steps)
		{
			const auto snapshot = simulation.make_snapshot(elapsed_ms);
			std::printf(
				"step=%llu time=%.7g dt=%.7g particles=%zu step_ms=%.3f prep_ms=%.3f "
				"max_rho=%.7g max_v=%.7g max_a=%.7g P=(%.8g,%.8g,%.8g) finite=%d\n",
				static_cast<unsigned long long>(snapshot.step),
				snapshot.total_time,
				snapshot.last_time_step,
				snapshot.particles.size(),
				snapshot.step_milliseconds,
				snapshot.preparation_milliseconds,
				snapshot.maximum_density,
				snapshot.maximum_speed,
				snapshot.maximum_acceleration,
				snapshot.momentum[0],
				snapshot.momentum[1],
				snapshot.momentum[2],
				snapshot.finite ? 1 : 0);
			if (!snapshot.finite)
				return 1;
		}
	}
	return 0;
}

int run_preparation_benchmark(int requested_particles, int requested_steps, float radius_in_spacings)
{
	const int side = static_cast<int>(std::ceil(std::cbrt(
		static_cast<double>(std::max(requested_particles, 1)))));
	std::vector<sph::Particle> particles;
	particles.reserve(requested_particles);
	for (int z = 0; z < side && particles.size() < static_cast<std::size_t>(requested_particles); ++z)
		for (int y = 0; y < side && particles.size() < static_cast<std::size_t>(requested_particles); ++y)
			for (int x = 0; x < side && particles.size() < static_cast<std::size_t>(requested_particles); ++x)
				particles.push_back(sph::Particle{
					.position = {
						x - (side - 1) * 0.5f,
						y - (side - 1) * 0.5f,
						z - (side - 1) * 0.5f},
					.velocity = {},
					.acceleration = {},
					.mass = 1.f,
					.smoothing_length = radius_in_spacings,
					.energy = 1.f,
					.cfl_time = 1.f});
	sph::SimulationConfig config;
	config.enable_gravity = requested_steps > 0;
	sph::Simulation3D simulation(std::move(particles), side + 4.f, config);
	const int runs = std::max(requested_steps, 1);
	double elapsed_ms = 0.;
	for (int run = 0; run < runs; ++run)
	{
		const auto begin = std::chrono::steady_clock::now();
		if (requested_steps > 0)
			(void)simulation.step();
		else
			simulation.prepare();
		elapsed_ms += std::chrono::duration<double, std::milli>(
			std::chrono::steady_clock::now() - begin).count();
	}
	const auto snapshot = simulation.make_snapshot();
	std::printf(
		"particles=%zu steps=%d radius=%.3f average_ms=%.3f prep_ms=%.3f "
		"octree_nodes=%zu occupied_cells=%zu\n",
		snapshot.particles.size(),
		requested_steps > 0 ? runs : 0,
		radius_in_spacings,
		elapsed_ms / runs,
		snapshot.preparation_milliseconds,
		snapshot.octree_nodes,
		snapshot.occupied_cells);
	return 0;
}

void glfw_error_callback(int error, const char* description)
{
	std::fprintf(stderr, "GLFW error %d: %s\n", error, description);
}

int run_gui(bool smoke_test = false)
{
	glfwSetErrorCallback(glfw_error_callback);
	if (!glfwInit())
		return 1;
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_SAMPLES, 4);
	if (smoke_test)
		glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
	GLFWwindow* window = glfwCreateWindow(1280, 800, "SPH Gas Cloud 3D", nullptr, nullptr);
	if (!window)
	{
		glfwTerminate();
		return 1;
	}
	glfwMakeContextCurrent(window);
	glfwSwapInterval(1);
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO();
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
	ImGui::StyleColorsDark();
	ImGuiStyle& style = ImGui::GetStyle();
	style.WindowRounding = 6.f;
	style.FrameRounding = 4.f;
	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init("#version 330 core");

	int exit_code = 0;
	try
	{
		sph::InitialConditions initial;
		if (smoke_test)
			initial.particle_count = 1000;
		sph::SimulationConfig simulation_config;
		app::VisualizationSettings visualization;
		app::Camera camera;
		camera.reset(initial.domain_size);
		app::ParticleRenderer renderer;
		sph::SimulationRunner runner(initial, simulation_config);
		int requested_particle_count = static_cast<int>(initial.particle_count);
		int requested_seed = static_cast<int>(initial.seed);
		const char* field_names[] = {
			"Density", "Energy", "Speed", "Acceleration",
			"Velocity X", "Velocity Y", "Velocity Z"};
		int selected_field = 0;

		while (!glfwWindowShouldClose(window))
		{
			glfwPollEvents();
			ImGui_ImplOpenGL3_NewFrame();
			ImGui_ImplGlfw_NewFrame();
			ImGui::NewFrame();

			ImGui::SetNextWindowSize(ImVec2(355.f, 650.f), ImGuiCond_FirstUseEver);
			ImGui::SetNextWindowPos(ImVec2(16.f, 16.f), ImGuiCond_FirstUseEver);
			ImGui::Begin("SPH Gas Cloud 3D");
			const bool running = runner.is_running();
			if (ImGui::Button(running ? "Pause" : "Run", ImVec2(90.f, 0.f)))
				runner.set_running(!running);
			ImGui::SameLine();
			if (ImGui::Button("Step", ImVec2(90.f, 0.f)))
				runner.request_single_step();
			ImGui::SameLine();
			if (ImGui::Button("Reset", ImVec2(90.f, 0.f)))
			{
				initial.particle_count = static_cast<std::size_t>(std::max(requested_particle_count, 1));
				initial.seed = static_cast<std::uint32_t>(std::max(requested_seed, 0));
				runner.request_reset(initial);
				camera.reset(initial.domain_size);
			}

			if (ImGui::CollapsingHeader("Initial conditions", ImGuiTreeNodeFlags_DefaultOpen))
			{
				ImGui::InputInt("Particles", &requested_particle_count, 1000, 10000);
				requested_particle_count = std::clamp(requested_particle_count, 1, 1000000);
				ImGui::InputInt("Seed", &requested_seed);
				ImGui::SliderFloat("Domain size", &initial.domain_size, 10.f, 500.f, "%.1f");
				ImGui::SliderFloat("Cloud radius", &initial.cloud_radius_fraction, 0.05f, 0.48f, "%.2f domain");
				ImGui::InputFloat("Total mass", &initial.total_mass, 1000.f, 10000.f, "%.1f");
				ImGui::SliderFloat("Angular speed", &initial.angular_speed, 0.f, 0.5f, "%.3f");
			}

			if (ImGui::CollapsingHeader("Simulation", ImGuiTreeNodeFlags_DefaultOpen))
			{
				ImGui::Checkbox("Hydrodynamics", &simulation_config.enable_hydrodynamics);
				ImGui::SameLine();
				ImGui::Checkbox("Gravity", &simulation_config.enable_gravity);
				ImGui::SliderFloat("Maximum dt", &simulation_config.maximum_time_step, 1e-5f, 0.02f, "%.6f", ImGuiSliderFlags_Logarithmic);
				ImGui::SliderFloat("Gravity G", &simulation_config.gravitational_constant, 0.f, 0.01f, "%.6f");
				ImGui::SliderFloat("Barnes-Hut theta", &simulation_config.barnes_hut_theta, 0.08f, 0.9f, "%.2f");
				ImGui::SliderInt("Desired neighbors", &simulation_config.desired_neighbors, 16, 128);
				ImGui::SliderFloat("Adiabatic gamma", &simulation_config.heat_capacity_ratio, 1.01f, 2.f, "%.3f");
				ImGui::SliderFloat("Polytropic exponent", &simulation_config.polytropic_exponent, 1.01f, 2.f, "%.3f");
				ImGui::SliderFloat("Polytropic strength", &simulation_config.polytropic_strength, 0.f, 1.f, "%.3f");
				runner.set_config(simulation_config);
			}

			if (ImGui::CollapsingHeader("Visualization", ImGuiTreeNodeFlags_DefaultOpen))
			{
				if (ImGui::Combo("Field", &selected_field, field_names, IM_ARRAYSIZE(field_names)))
					visualization.field = static_cast<sph::Field>(selected_field);
				ImGui::SliderFloat("Brightness", &visualization.brightness, 0.05f, 5.f, "%.2f", ImGuiSliderFlags_Logarithmic);
				ImGui::SliderFloat("Point size", &visualization.point_size, 1.f, 16.f, "%.1f px");
				ImGui::Checkbox("Logarithmic color scale", &visualization.logarithmic_scale);
				ImGui::Checkbox("Show initial domain", &visualization.show_domain_box);
				ImGui::ColorEdit3("Background", visualization.background);
				if (ImGui::Button("Reset camera"))
					camera.reset(initial.domain_size);
			}

			const std::shared_ptr<const sph::SimulationSnapshot> snapshot = runner.snapshot();
			if (snapshot && ImGui::CollapsingHeader("Telemetry", ImGuiTreeNodeFlags_DefaultOpen))
			{
				ImGui::Text("Particles: %zu", snapshot->particles.size());
				ImGui::Text("Step: %llu   Time: %.6g", static_cast<unsigned long long>(snapshot->step), snapshot->total_time);
				ImGui::Text("dt: %.4g", snapshot->last_time_step);
				ImGui::Text("Step: %.2f ms   Prepare: %.2f ms", snapshot->step_milliseconds, snapshot->preparation_milliseconds);
				ImGui::Text("Octree nodes: %zu", snapshot->octree_nodes);
				ImGui::Text("Occupied hash cells: %zu", snapshot->occupied_cells);
				ImGui::Text("Max density: %.5g", snapshot->maximum_density);
				ImGui::Text("Max speed: %.5g", snapshot->maximum_speed);
				ImGui::Text("Max acceleration: %.5g", snapshot->maximum_acceleration);
				if (!snapshot->finite)
					ImGui::TextColored(ImVec4(1.f, 0.25f, 0.2f, 1.f), "Non-finite state detected");
			}
			ImGui::Separator();
			ImGui::TextDisabled("LMB: orbit   RMB/MMB: pan   Wheel: zoom   F: reset view");
			ImGui::End();

			if (!io.WantCaptureMouse)
			{
				if (ImGui::IsMouseDragging(ImGuiMouseButton_Left, 0.f))
					camera.orbit(io.MouseDelta.x, io.MouseDelta.y);
				if (ImGui::IsMouseDragging(ImGuiMouseButton_Right, 0.f) ||
					ImGui::IsMouseDragging(ImGuiMouseButton_Middle, 0.f))
					camera.pan(io.MouseDelta.x, io.MouseDelta.y, io.DisplaySize.y);
				if (io.MouseWheel != 0.f)
					camera.zoom(io.MouseWheel);
			}
			if (!io.WantCaptureKeyboard && ImGui::IsKeyPressed(ImGuiKey_F))
				camera.reset(initial.domain_size);

			ImGui::Render();
			int framebuffer_width = 0;
			int framebuffer_height = 0;
			glfwGetFramebufferSize(window, &framebuffer_width, &framebuffer_height);
			glViewport(0, 0, framebuffer_width, framebuffer_height);
			glClearColor(
				visualization.background[0],
				visualization.background[1],
				visualization.background[2],
				1.f);
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
			if (snapshot)
				renderer.draw(
					*snapshot,
					camera,
					visualization,
					framebuffer_width,
					framebuffer_height,
					initial.domain_size);
			ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
			glfwSwapBuffers(window);
			if (smoke_test)
				glfwSetWindowShouldClose(window, GLFW_TRUE);
		}
	}
	catch (const std::exception& exception)
	{
		std::fprintf(stderr, "Fatal application error: %s\n", exception.what());
		exit_code = 1;
	}

	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
	glfwDestroyWindow(window);
	glfwTerminate();
	return exit_code;
}

} // namespace

int main(int argc, char** argv)
{
	if (argc > 1 && std::string(argv[1]) == "--self-test")
		return run_numerical_self_tests();
	if (argc > 1 && std::string(argv[1]) == "--headless")
	{
		const int steps = argc > 2 ? std::max(std::atoi(argv[2]), 1) : 100;
		const std::uint32_t seed = argc > 3 ?
			static_cast<std::uint32_t>(std::strtoul(argv[3], nullptr, 10)) : 1u;
		const std::size_t particles = argc > 4 ?
			static_cast<std::size_t>(std::max(std::atoi(argv[4]), 1)) : 1000;
		return run_headless(steps, seed, particles);
	}
	if (argc > 1 && std::string(argv[1]) == "--benchmark-preparation")
	{
		const int particles = argc > 2 ? std::max(std::atoi(argv[2]), 1) : 10000;
		const int steps = argc > 3 ? std::atoi(argv[3]) : 0;
		const float radius = argc > 4 ?
			std::max(static_cast<float>(std::atof(argv[4])), 0.01f) : 2.5f;
		return run_preparation_benchmark(particles, steps, radius);
	}
	if (argc > 1 && std::string(argv[1]) == "--gui-smoke-test")
		return run_gui(true);
	return run_gui();
}

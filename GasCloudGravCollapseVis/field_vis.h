#pragma once
#include <vector>
#include <utility>

inline std::tuple<float, float, float> HSVtoRGB(const float& fH, const float& fS, const float& fV) {
	float fR;
	float fG;
	float fB;
	float fC = fV * fS; // Chroma
	float fHPrime = fmod(fH / 60.0, 6);
	float fX = fC * (1 - fabs(fmod(fHPrime, 2) - 1));
	float fM = fV - fC;

	if (0 <= fHPrime && fHPrime < 1) {
		fR = fC;
		fG = fX;
		fB = 0;
	}
	else if (1 <= fHPrime && fHPrime < 2) {
		fR = fX;
		fG = fC;
		fB = 0;
	}
	else if (2 <= fHPrime && fHPrime < 3) {
		fR = 0;
		fG = fC;
		fB = fX;
	}
	else if (3 <= fHPrime && fHPrime < 4) {
		fR = 0;
		fG = fX;
		fB = fC;
	}
	else if (4 <= fHPrime && fHPrime < 5) {
		fR = fX;
		fG = 0;
		fB = fC;
	}
	else if (5 <= fHPrime && fHPrime < 6) {
		fR = fC;
		fG = 0;
		fB = fX;
	}
	else {
		fR = 0;
		fG = 0;
		fB = 0;
	}
	fR += fM;
	fG += fM;
	fB += fM;
	return { fR, fG, fB };
}

inline std::tuple<float, float, float> get_color(float value) {
	constexpr float delay = 15;
	value = value / delay;
	if (value > 0) {
		if (value <= 1)
			return HSVtoRGB(120, 1, value);
		if (value <= 2)
			return HSVtoRGB(120 - 120 * (value - 1), 1, 1);
		if (value <= 3)
			return HSVtoRGB(0, 1 - (value - 2), 1);
		else
			return HSVtoRGB(0, 0, 1);
	}
	else {
		value = -value;
		if (value <= 1)
			return HSVtoRGB(300, 1, value);
		if (value <= 2)
			return HSVtoRGB(300 - (value - 1) * 120, 1, 1);
		if (value <= 3)
			return HSVtoRGB(180, 1 - (value - 2), 1);
		else
			return HSVtoRGB(0, 0, 1);
	}
}

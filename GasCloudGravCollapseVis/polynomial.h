#pragma once
#include <string>
#include <vector>
#include <iostream>
#include <utility>
#include <set>
#include <algorithm>
#include <iomanip>

struct polynomial {
	std::vector<double> ar;
	polynomial(size_t d, double val) {
		ar = std::vector<double>(d+1, 0);
		ar.back() = val;
	}
	polynomial(double val=1) : polynomial(0,val) {}
	polynomial(const std::initializer_list<double> &dblist) {
		ar = dblist; 
		recapture_cur_degree();
	}
	polynomial operator=(const polynomial& bp) {
		ar = bp.ar;
		return *this;
	}
	inline double& at(size_t deg) {
		if (deg <= degree())
			return ar[deg];
		else {
			ar.resize(deg+1,0);
			return ar[deg];
		}
	}
	inline double& operator[](size_t deg) {
		return at(deg);
	}
	inline double at(size_t deg) const {
		if (deg <= degree())
			return ar[deg];
		else
			return 0;
	}
	inline double operator[](size_t deg) const {
		return at(deg);
	}
	inline size_t degree() const {
		return ar.size() - 1;
	}
	inline void recapture_cur_degree() {
		while (abs(ar.back()) < DBL_EPSILON && degree() > 0)
			ar.pop_back();
	}
	inline polynomial operator+(const polynomial& bp) const {
		polynomial res = *this;
		for (size_t d = 0; d <= bp.degree(); d++)
			res.at(d) += bp.at(d);
		res.recapture_cur_degree();
		return res;
	}
	inline polynomial operator+=(const polynomial& bp) {
		return (*this = *this + bp);
	}
	inline polynomial operator+(double d) const {
		polynomial res = *this;
		res.at(0) += d;
		return res;
	}
	inline polynomial operator+=(double d) {
		return (*this = *this + d);
	}
	inline polynomial operator-(const polynomial& bp) const {
		polynomial res = *this;
		for (size_t d = 0; d <= bp.degree(); d++)
			res.at(d) -= bp.at(d);
		res.recapture_cur_degree();
		return res;
	}
	inline polynomial operator-=(const polynomial& bp) {
		return (*this = *this - bp);
	}
	inline polynomial operator-(double d) const {
		polynomial res = *this;
		res.at(0) -= d;
		return res;
	}
	inline polynomial operator-=(double d) {
		return (*this = *this - d);
	}
	inline polynomial operator*(const polynomial& bp) const {
		polynomial res = {0};
		for (size_t bpd = 0; bpd <= bp.degree(); bpd++)
			for (size_t d = 0; d <= degree(); d++)
				res.at(bpd+d) += at(d)*bp.at(bpd);
		res.recapture_cur_degree();
		return res;
	}
	inline polynomial operator*=(const polynomial& bp) {
		return (*this = *this * bp);
	}
	inline polynomial operator*(double d) const {
		polynomial res = *this;
		for (size_t deg = 0; deg <= degree(); deg++)
			res.at(deg) *= d;
		return res;
	}
	inline polynomial operator*=(double d) {
		return (*this = *this * d);
	}
	/*inline polynomial operator/(const polynomial& bp) const {
		if (bp.degree() > degree())
			return { 0 };
		polynomial res = *this, tmod = { 0 };
		for (intptr_t bpd = bp.degree(); bpd >= 0; bpd++) {
			//todo .. finish it
		}
		res.recapture_cur_degree();
		return res;
	}*/
	inline polynomial operator/(double d) const {
		return (*this * (1 / d));
	}
	inline polynomial operator/=(double d) {
		return (*this = *this / d);
	}
	inline double operator()(double num) const {
		double sum = 0;
		for (size_t d = 0; d < degree(); d++)
			sum += pow(num, d)*at(d);
		return sum;
	}
	inline polynomial derivative() const {
		polynomial res = *this; 
		for (intptr_t bpd = bp.degree(); bpd >= 0; bpd++) {
			res.at(bpd) = (bpd + 1) * res.at(bpd + 1);
		}
		return res;
	}
};
inline polynomial operator*(double d, const polynomial& bp) {
	return bp * d;
}
inline polynomial operator+(double d, const polynomial& bp) {
	return bp + d;
}
inline polynomial operator-(double d, const polynomial& bp) {
	return (bp*(-1)) + d;
}

inline std::istream& operator>>(std::istream& in, polynomial& M) {
	size_t d;
	in >> d;
	M = polynomial(d);
	for (auto& l : M.ar)
		in >> l;
	return in;
}
inline std::ostream& operator<<(std::ostream& out, const polynomial& M) {
	for (auto d = M.ar.begin(); d != M.ar.end(); d++)
		out << std::setprecision(3) << *d << " ";
	return out;
}

inline std::ostream& operator<<(std::ostream& out, const polynomial_matrix& M) {
	for (auto&& l : M) {
		for (auto&& d : l) {
			out << d << "\t";
		}
		out << '\n';
	}
	return out;
}

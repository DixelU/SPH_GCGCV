#pragma once
#include <cstdio>
#include <array>
#include <cmath>
#include <ostream>
#include <initializer_list>

#define _x(p) ((p)[0])
#define _y(p) ((p)[1])
#define _z(p) ((p)[2])

template<size_t dims>
struct Point {
	std::array<double, dims> pt;
	Point() {
		for (int i = 0; i < dims; i++)
			pt[i] = 0;
	}
	Point(const Point<dims> &P) {
		for (int i = 0; i < dims; i++)
			pt[i] = P.pt[i];
	}
	Point(const std::initializer_list<double>& il_d) {
		std::initializer_list<double>::iterator y = il_d.begin();
		for (size_t i = 0; i < dims && y != il_d.end(); i++, y++)
			pt[i] = *y;
	}
	Point(const std::initializer_list<int>& il) {
		std::initializer_list<int>::iterator y = il.begin();
		for (size_t i = 0; i < dims && y != il.end(); i++, y++)
			pt[i] = *y;
	}
	Point(const std::vector<double>& il_d) {
		std::vector<double>::const_iterator y = il_d.cbegin();
		for (size_t i = 0; i < dims && y != il_d.end(); i++, y++)
			pt[i] = *y;
	}
	Point(const std::vector<int>& il) {
		std::vector<int>::const_iterator y = il.cbegin();
		for (size_t i = 0; i < dims && y != il.end(); i++, y++)
			pt[i] = *y;
	}
	inline double norma2() const {
		double sum = 0;
		for (int i = 0; i < dims; i++)
			sum += pt[i] * pt[i];
		return sum;
	}
	inline double norma() const {
		return std::sqrt(norma2());
	}
	inline size_t get_dims() const {
		return dims;
	}
	inline double& operator[](size_t D) {
		return pt[D];
	}
	inline const double& operator[](size_t D) const {
		return pt[D];
	}
	inline void swap(Point<dims> &P) {
		pt.swap(P.pt);
	}
	inline Point<dims> operator+(const Point<dims> &P) const {
		Point<dims> N;
		for (size_t i = 0; i < dims; i++)
			N[i] = pt[i] + P[i];
		return N;
	}
	inline Point<dims> operator+=(const Point<dims> &P) {
		for (size_t i = 0; i < dims; i++)
			pt[i] += P[i];
		return *this;
	}
	inline Point<dims> operator-(const Point<dims> &P) const {
		Point<dims> N;
		for (size_t i = 0; i < dims; i++)
			N[i] = pt[i] - P[i];
		return N;
	}
	inline Point<dims> operator-=(const Point<dims> &P) {
		for (size_t i = 0; i < dims; i++)
			pt[i] -= P[i];
		return *this;
	}
	inline Point<dims> operator*(double M) const {
		Point<dims> N;
		for (size_t i = 0; i < dims; i++)
			N[i] = pt[i] * M;
		return N;
	}
	inline Point<dims> operator*=(double M) {
		for (size_t i = 0; i < dims; i++)
			pt[i] *= M;
		return *this;
	}
	inline Point<dims> operator/(double M) const {
		Point<dims> N;
		for (size_t i = 0; i < dims; i++)
			N[i] = pt[i] / M;
		return N;
	}
	inline Point<dims> operator/=(double M) {
		return ((*this)*=(1. / M));
	}
	inline double operator*(const Point<dims> &P) const {
		double sum = 0;
		for (size_t i = 0; i < dims; i++)
			sum += pt[i] * P[i];
		return sum;
	}
	inline Point<dims> operator-() {
		for (size_t i = 0; i < dims; i++)
			pt[i] = 0 - pt[i];
		return *this;
	}
	inline bool operator>(const Point<dims>& P) const {
		for (size_t i = 0; i < dims; i++)
			if (pt[i] <= P.pt[i])
				return false;
		return true;
	}
	inline bool operator>=(const Point<dims>& P) const {
		for (size_t i = 0; i < dims; i++)
			if (pt[i] < P.pt[i])
				return false;
		return true;
	}
	inline bool operator<(const Point<dims>& P) const {
		for (size_t i = 0; i < dims; i++)
			if (pt[i] >= P.pt[i])
				return false;
		return true;
	}
	inline bool operator<=(const Point<dims>& P) const {
		for (size_t i = 0; i < dims; i++)
			if (pt[i] > P.pt[i])
				return false;
		return true;
	}
};

template<size_t dims>
std::ostream& operator<<(std::ostream& os, const Point<dims> &P) {
	os << "(";
	for (size_t i = 0; i < dims; i++) {
		os << P[i];
		if (i != dims - 1)
			os << ",";
	}
	os << ")";
	return os;
}

template<size_t dims>
inline Point<dims> operator*(double M, Point<dims> P) {
	Point<dims> N;
	for (size_t i = 0; i < dims; i++)
		N[i] = P[i] * M;
	return N;
}

template<size_t dims>
struct sq_matrix {
	double utilisation = 0;
	std::array<Point<dims>, dims> ar;
	sq_matrix() {
		for (size_t i = 0; i < dims; i++) 
			ar[i] = Point<dims>();
	}
	sq_matrix(double E_num) {
		for (size_t i = 0; i < dims; i++) {
			ar[i] = Point<dims>();
			ar[i][i] = E_num;
		}
	}
	sq_matrix(std::initializer_list<Point<dims>> IL) {
		size_t id = 0;
		for (auto &&p : IL) {
			if (id == dims)
				break;
			ar[id] = p;
			id++;
		}
	}
	inline Point<dims> operator*(const Point<dims> &p) const {
		Point<dims> T;
		for (int i = 0; i < dims; i++) {
			T[i] = ar[i] * p;
		}
		return T;
	}
	inline const Point<dims>& operator[](size_t i) const {
		return ar[i];
	}
	inline Point<dims>& operator[](size_t i) {
		return ar[i];
	}
	inline double& at(size_t point_id, size_t coordinate) {
		if (point_id < dims && coordinate < dims) {
			return ar[point_id][coordinate];
		}
		else return utilisation;
	}
	inline const double& at(size_t point_id, size_t coordinate) const {
		if (point_id < dims && coordinate < dims) {
			return ar[point_id][coordinate];
		}
		else return utilisation;
	}
	inline sq_matrix<dims> operator*(const sq_matrix<dims> &M) const {
		sq_matrix<dims> P;
		for (size_t y = 0; y < dims; y++) {
			for (size_t x = 0; x < dims; x++) {
				for (size_t i = 0; i < dims; i++) {
					P[y][x] += ar[y][i] * M[i][x];
				}
			}
		}
		return P;
	}
	inline sq_matrix<dims> operator+(const sq_matrix<dims> &M) const {
		sq_matrix<dims> P;
		for (size_t y = 0; y < dims; y++) {
			P[y] = M[y] + ar[y];
		}
		return P;
	}
	inline sq_matrix<dims> operator-(const sq_matrix<dims> &M) const {
		sq_matrix<dims> P;
		for (size_t y = 0; y < dims; y++) {
			P[y] = ar[y] - M[y];
		}
		return P;
	}
	sq_matrix<dims> operator*(const double m) const {
		sq_matrix<dims> P;
		for (size_t y = 0; y < dims; y++) {
			P.ar[y] = ar[y] * m;
		}
		return P;
	}
	sq_matrix<dims> operator/(const double d) const {
		return (*this)*(1 / d);
	}
	inline sq_matrix<dims> inverse() const {
		double temp = 0, max = 0, mul = 0;
		size_t id = 0;
		sq_matrix<dims> E(1), A(*this);
		for (size_t step = 0; step < dims; step++){
			id = step;
			max = 0;
			for (size_t coid = id; coid < dims; coid++) {
				if (max < abs(A.at(coid, step))) {
					max = abs(A.at(coid, step));
					id = coid;
				}
			}
			//std::cout << A << E << std::endl;
			if (abs(max) <= DBL_EPSILON)
				return sq_matrix<dims>(1);
			if (id != step)
				A.ar[id].swap(A.ar[step]);
			for (size_t sum_id = 0; sum_id < dims; sum_id++) {
				if (sum_id == step || A.at(sum_id,step) <= DBL_EPSILON)
					continue;
				//std::cout << "sum\n" << A << E << std::endl;
				mul = A.at(sum_id, step) / A.at(step,step);
				A[sum_id] -= (A[step] * mul);
				E[sum_id] -= (E[step] * mul);
			}
			mul = A.at(step, step);
			A[step] /= mul;
			E[step] /= mul;
		}
		return E;
	}
};

template<size_t dims>
inline sq_matrix<dims> cross_prod(const Point<dims> &a, const Point<dims> &b) {
	sq_matrix<dims> sm;
	for (size_t y = 0; y < dims; y++) {
		for (size_t x = 0; x < dims; x++) {
			sm.at(y, x) = a[y] * b[x];
		}
	}
	return sm;
}

template<size_t dims>
inline std::ostream& operator<<(std::ostream& in, const sq_matrix<dims>& M) {
	for (size_t y = 0; y < dims; y++) {
		for (size_t x = 0; x < dims; x++) {
			in << floor(abs(M.at(y, x))*100.) / 100. * (M.at(y, x)<0?-1:1) << "\t";
		}
		in << "\n";
	}
	return in;
}

template<size_t dims>
inline sq_matrix<dims> operator*(const double n, sq_matrix<dims> M) {
	return M * n;
}
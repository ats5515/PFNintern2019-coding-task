#pragma once

#include <algorithm>
#include <iostream>
#include <vector>
#include <cassert>

//グラフクラスの実装
class Graph {

public:
	//隣接リストを用いた。
	std::vector<std::vector<int> > adj;
	
	//入力
	friend std::istream& operator>>(std::istream& is, Graph& graph) {
		graph.adj.clear();
		int N; is >> N;
		graph.adj.resize(N);
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) {
				int edge; is >> edge;
				if (edge == 1) {
					graph.adj[i].push_back(j);
				}
			}
		}

		return is;
	}

	//出力
	friend std::ostream& operator<<(std::ostream& os, Graph& graph) {
		int N = graph.adj.size();
		for (int i = 0; i < N; i++) {
			for (int j : graph.adj[i]) {
				os << i << " -> " << j << std::endl;
			}
		}
		return os;
	}
};
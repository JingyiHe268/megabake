"""Check Mirage's unmodified allocation core against an interval oracle.

This checks host allocation on supplied lifetimes. It does not check Mirage's
lifetime analysis, asynchronous GPU uses, or megakernel composition.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


REVISION = "f9eb70c254acefc9f3667b2a973d0dcf25471fce"
SOURCE_HASH = "4b1400e4aa90cae8411c648002c23b6de899301fbbeedb982e4282fd29921657"
ROOT = Path(__file__).resolve().parents[2]

PREAMBLE = r"""
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <set>
#include <unordered_map>
#include <vector>
using std::vector;
using sguid_t = std::size_t;
"""

CHECKS = r"""
using namespace memory_planner;

void check(vector<TensorDecl> const &ds, AllocResult const &r) {
  assert(r.addrs.size() == ds.size());
  std::size_t total = 0;
  for (auto const &a : ds) {
    total += a.phy_size;
    auto start = r.addrs.at(a.sguid);
    assert(start % 128 == 0);
    assert(start + a.phy_size <= r.peak_memory_usage);
    for (auto const &b : ds) {
      if (a.sguid == b.sguid) continue;
      bool live_together = a.alloc_time < b.free_time &&
                           b.alloc_time < a.free_time;
      if (live_together) {
        auto other = r.addrs.at(b.sguid);
        assert(start + a.phy_size <= other ||
               other + b.phy_size <= start);
      }
    }
  }
  assert(r.peak_memory_usage <= total);
  for (int time = 0; time < 65; ++time) {
    std::size_t live_bytes = 0;
    for (auto const &a : ds)
      if (a.alloc_time <= time && time < a.free_time)
        live_bytes += a.phy_size;
    assert(live_bytes <= r.peak_memory_usage);
  }
}

int main() {
  std::mt19937 gen(20261009);
  constexpr int cases = 2000;
  for (int i = 0; i < cases; ++i) {
    vector<TensorDecl> ds;
    int count = 1 + gen() % 40;
    for (int j = 0; j < count; ++j) {
      int begin = gen() % 32;
      int end = begin + 1 + gen() % 32;
      ds.push_back({std::size_t(j), 128 * (1 + gen() % 32), begin, end});
    }
    std::shuffle(ds.begin(), ds.end(), gen);
    vector<std::shared_ptr<AbstractMemoryPlanner>> ps = {
      std::make_shared<FirstFitMemoryPlanner>(),
      std::make_shared<BestFitMemoryPlanner>(),
      std::make_shared<WorseFitMemoryPlanner>()};
    std::size_t best = std::numeric_limits<std::size_t>::max();
    for (auto const &p : ps) {
      for (auto const &d : ds) p->declare_tensor(d);
      auto result = p->get_allocation();
      check(ds, result);
      best = std::min(best, result.peak_memory_usage);
    }
    auto result = plan_memory(ds);
    check(ds, result);
    assert(result.peak_memory_usage == best);
  }
  auto reuse = plan_memory({{0, 128, 0, 1}, {1, 128, 1, 2}});
  assert(reuse.peak_memory_usage == 128);
  auto overlap = plan_memory({{0, 128, 0, 2}, {1, 128, 1, 3}});
  assert(overlap.peak_memory_usage == 256);
  std::cout << "{\"random_cases\":" << cases
            << ",\"policies_per_case\":4,\"boundary_cases\":2,"
            << "\"passed\":true}\n";
}
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    clone = ROOT / "agent_space/mirage"
    source = clone / "src/transpiler/plan_stensor_memory.cc"
    revision = subprocess.check_output(
        ["git", "-C", str(clone), "rev-parse", "HEAD"], text=True
    ).strip()
    assert revision == REVISION, revision
    data = source.read_bytes()
    assert hashlib.sha256(data).hexdigest() == SOURCE_HASH
    text = data.decode()
    start, end = "namespace memory_planner {", "} // namespace memory_planner"
    assert text.count(start) == text.count(end) == 1
    core = text[text.index(start):text.index(end) + len(end)]
    with tempfile.TemporaryDirectory(prefix="megabake-memory-reuse-") as tmp:
        cpp, exe = Path(tmp) / "check.cc", Path(tmp) / "check"
        cpp.write_text(PREAMBLE + core + CHECKS)
        compiler = subprocess.check_output(["g++", "--version"], text=True)
        subprocess.run(["g++", "-std=c++17", "-O2", str(cpp), "-o", str(exe)],
                       check=True)
        result = json.loads(subprocess.check_output([str(exe)], text=True))
    result.update(revision=revision, source_sha256=SOURCE_HASH,
                  compiler=compiler.splitlines()[0], seed=20261009,
                  limits="Host allocation only; valid nonzero sizes padded to 128 bytes; supplied finite half-open lifetimes.")
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

#include "mcts_fast.hpp"
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

PYBIND11_MODULE(mcts_cpp, m, py::mod_gil_not_used()) {
    m.doc() = "High performance MCTS module";
    py::class_<MCTS>(m, "MCTS")
        .def(py::init<>())
        .def("gather_batch", &MCTS::gatherBatch)
        .def("get_root_visits", &MCTS::getRootVisits)
        .def("setRootState", &MCTS::setRootState)
        .def("changeRoot", &MCTS::changeRoot)
        .def("get_root_value", &MCTS::getRootValue)
        .def("expand_backup", &MCTS::expandBackup);
}
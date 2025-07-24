"""
Benchmarking components package.

This package provides comprehensive GPU vs CPU performance benchmarking
capabilities for the pendulum AI control system, including real hardware
acceleration testing and detailed performance reporting.

Key Components:
- GPU vs CPU benchmark execution with real MAX Engine DeviceContext API
- Comprehensive performance metrics collection and analysis
- Detailed technical report generation with system information
- Hardware acceleration validation and performance comparison

You can import these APIs from the `src.benchmarks` package. For example:

```mojo
from src.benchmarks import RealGPUCPUBenchmark, BenchmarkReportGenerator
from src.benchmarks import create_real_benchmark_system, generate_real_gpu_report
```
"""

# Import key public APIs to make them accessible at package level
from src.benchmarks.gpu_cpu_benchmark import (
    RealGPUCPUBenchmark,
    BenchmarkResult,
    create_real_benchmark_system,
    run_real_gpu_benchmark,
    ComputeMode_AUTO,
    ComputeMode_GPU_ONLY,
    ComputeMode_CPU_ONLY,
    ComputeMode_HYBRID,
)

from src.benchmarks.report_generator import (
    BenchmarkReportGenerator,
    BenchmarkMetrics,
    SystemInfo,
    create_benchmark_report,
    generate_real_gpu_report,
    generate_sample_report,
    collect_real_gpu_metrics,
)

---
layout: default
parent: Tutorials
nav_order: 8
---

GPUDirect Use
======================

# CuFile Configuration Settings
The CuFile library has a configuration file located by default at `/etc/cufile.json`. Users might find that the default configuration may not be suited for their workload/application needs and instead need to make modifications. To do this first make a local copy at some path.
```bash
cp /etc/cufile.json /some/path/cufile.json`
```
Then the `CUFILE_ENV_PATH_JSON` environment variable must be updated to point to the path of the new configuration file.
```bash
export CUFILE_ENV_PATH_JSON="/some/path/cufile.json"
```
A list of the parameters and their values are described in the [GPUDirect Storage Benchmarking and Configuration Guide](https://docs.nvidia.com/gpudirect-storage/configuration-guide/index.html#gpudirect-storage-parameters) Make and save any changes necessary. 

# Verify GPUDirect Storage is Supported
GPUDirect storage (GDS) is an experimental and evolving technology. There may comes times when users suspect GDS is not working as intended. A first check is to investigate whether the GDS associated libraries are loaded correctly. This can be done by calling `gdscheck.py -p`.
```bash
[strugf@gpu0206 ~]$ /usr/local/cuda-12.6/gds/tools/gdscheck.py -p
 GDS release version: 1.11.1.6
 nvidia_fs version:  2.25 libcufile version: 2.12
 Platform: x86_64
 ============
 ENVIRONMENT:
 ============
 =====================
 DRIVER CONFIGURATION:
 =====================
 NVMe               : Supported
 NVMeOF             : Unsupported
 SCSI               : Unsupported
 ScaleFlux CSD      : Unsupported
 NVMesh             : Unsupported
 DDN EXAScaler      : Unsupported
 IBM Spectrum Scale : Unsupported
 NFS                : Unsupported
 BeeGFS             : Unsupported
 WekaFS             : Unsupported
 Userspace RDMA     : Unsupported
 --Mellanox PeerDirect : Enabled
 --rdma library        : Not Loaded (libcufile_rdma.so)
 --rdma devices        : Not configured
 --rdma_device_status  : Up: 0 Down: 0
 =====================
 CUFILE CONFIGURATION:
 =====================
 properties.use_compat_mode : true
 properties.force_compat_mode : false
 properties.gds_rdma_write_support : true
 properties.use_poll_mode : false
 properties.poll_mode_max_size_kb : 4
 properties.max_batch_io_size : 128
 properties.max_batch_io_timeout_msecs : 5
 properties.max_direct_io_size_kb : 16384
 properties.max_device_cache_size_kb : 131072
 properties.max_device_pinned_mem_size_kb : 33554432
 properties.posix_pool_slab_size_kb : 4 1024 16384 
 properties.posix_pool_slab_count : 128 64 32 
 properties.rdma_peer_affinity_policy : RoundRobin
 properties.rdma_dynamic_routing : 0
 fs.generic.posix_unaligned_writes : false
 fs.lustre.posix_gds_min_kb: 0
 fs.beegfs.posix_gds_min_kb: 0
 fs.weka.rdma_write_support: false
 fs.gpfs.gds_write_support: false
 profile.nvtx : false
 profile.cufile_stats : 0
 miscellaneous.api_check_aggressive : false
 execution.max_io_threads : 4
 execution.max_io_queue_depth : 128
 execution.parallel_io : true
 execution.min_io_threshold_size_kb : 8192
 execution.max_request_parallelism : 4
 properties.force_odirect_mode : false
 properties.prefer_iouring : false
 =========
 GPU INFO:
 =========
 GPU index 0 NVIDIA RTX A6000 bar:1 bar size (MiB):256 supports GDS, IOMMU State: Pass-through or Enabled
 GPU index 1 NVIDIA RTX A6000 bar:1 bar size (MiB):256 supports GDS, IOMMU State: Pass-through or Enabled
 GPU index 2 NVIDIA RTX A6000 bar:1 bar size (MiB):256 supports GDS, IOMMU State: Pass-through or Enabled
 GPU index 3 NVIDIA RTX A6000 bar:1 bar size (MiB):256 supports GDS, IOMMU State: Pass-through or Enabled
 ==============
 PLATFORM INFO:
 ==============
 IOMMU: Pass-through or enabled
 WARN: GDS is not guaranteed to work functionally or in a performant way with iommu=on/pt
 Nvidia Driver Info Status: Supported(Nvidia Open Driver Installed)
 Cuda Driver Version Installed:  12060
 Platform: MZ33-AR0-000, Arch: x86_64(Linux 5.14.0-503.33.1.el9_5.x86_64)
 Platform verification succeeded
```
Let's read through some of the most important lines of the output.
```bash
GDS release version: 1.11.1.6
 nvidia_fs version:  2.25 libcufile version: 2.12
```
This informs us that `libcufile` has been loaded and the `nvidia-fs` driver responsible for memory mapping between storage and the GPU is installed.
```bash
=====================
 DRIVER CONFIGURATION:
 =====================
 NVMe               : Supported
 NVMeOF             : Unsupported
 SCSI               : Unsupported
 ScaleFlux CSD      : Unsupported
 NVMesh             : Unsupported
 DDN EXAScaler      : Unsupported
 IBM Spectrum Scale : Unsupported
 NFS                : Unsupported
 BeeGFS             : Unsupported
 WekaFS             : Unsupported
 Userspace RDMA     : Unsupported
 --Mellanox PeerDirect : Enabled
 --rdma library        : Not Loaded (libcufile_rdma.so)
 --rdma devices        : Not configured
 --rdma_device_status  : Up: 0 Down: 0
```
We see `NVMe: Supported` which informs that GDS is currently configured to work for NVMe drives, and all other storage types are not properly configured as apparent from the `Unsupported` flag.

It is also possible to verify GDS is working by setting the `cufile.json` parameter `logging:level` to `TRACE`. When running a process that makes CuFile API calls, a `cufile.log` will be created in the directory the process is called from. The log is highly verbose, but it is possible to see the GDS transfers.
```bash
25-06-2025 13:24:20:556 [pid=1535029 tid=1535093] DEBUG cufio_core:2716 gds path taken with ODIRECT fd: 4 
25-06-2025 13:24:20:556 [pid=1535029 tid=1535093] DEBUG 0:1709 nvfs_io_submit file_offset 0 size 4194304 gpu_offset 0 nvbuf 0x7fae3d1bb700 is_unaligned 0 
25-06-2025 13:24:20:556 [pid=1535029 tid=1535093] DEBUG 0:461 current cuda context present 
25-06-2025 13:24:20:556 [pid=1535029 tid=1535093] DEBUG 0:472 Allocate buffer of size 1048576 on GPU 0 PCI-Group 0 
25-06-2025 13:24:20:556 [pid=1535029 tid=1535093] DEBUG 0:389 Bounce buffer 140380150431744 GPU page aligned 
25-06-2025 13:24:20:556 [pid=1535029 tid=1535093] DEBUG 0:481 Buffer from aligned alloc, dptr 0x7faccd000000 aligned_dptr 0x7faccd000000 size 1048576 
25-06-2025 13:24:20:556 [pid=1535029 tid=1535093] DEBUG 0:335 map buf 0x7faccd000000 Size 1048576 sbuf_size 1048576 pin_gpu_memory 1 
25-06-2025 13:24:20:556 [pid=1535029 tid=1535093] DEBUG 0:336 map buf 0x7faccd000000 bounce-buffer 1 groupId 0 
25-06-2025 13:24:20:556 [pid=1535029 tid=1535093] DEBUG 0:1093 Total usage 0 Max Usage 33554432 
25-06-2025 13:24:20:558 [pid=1535029 tid=1535093] DEBUG 0:487 MAP gpu index : 0 bdf: 0 1 0 0 
25-06-2025 13:24:20:559 [pid=1535029 tid=1535093] DEBUG 0:507 Buffer allocation and map success on GPU: 0 
25-06-2025 13:24:20:559 [pid=1535029 tid=1535093] DEBUG 0:842 Bounce-buffer allocated from PCI-Group: 0 GPU: 0
```
# KvikIO: Making cuFile API Calls in Python
KvikIO is a python library for high performance file IO. It provides python bindings to cuFile which enables the use of GDS in python applications. KvikIO comes with its own runtime properties that can be set globally or with a context manager.
```python
# Set the property globally.
kvikio.defaults.set({"prop1": value1, "prop2": value2})

# Set the property with a context manager.
# The property automatically reverts to its old value
# after leaving the `with` block.
with kvikio.defaults.set({"prop1": value1, "prop2": value2}):
    ...
```
The list of properties and their allowed values can be found in the [kvikio python documentation](https://docs.rapids.ai/api/kvikio/nightly/api/#kvikio.defaults.set).

Below is a minimal example of reading a file using kvikio's cuFile bindings.
```python
import kvikio, cupy
import kvikio.defaults

filepath = "/mnt/nvme/temp.file"
MiB = 1048576
size = 10 * MiB

with kvikio.defaults.set({"compat_mode": False}):
	with kvikio.CuFile(filepath, "rb") as f:
		buf = cupy.empty(size, dtype = "b")
		fut = f.pread(buf) # pread is non-blocking and returns a future that must be 
						   # waited upon
		fut.get()

print(buf)
print("Done!")
```
GDS is only available when reading from configured drives. At ACCRE, the `/mnt/nvme*` drives support GDS reading and writing. We set `compat_mode` to `False` to enforce cuFile I/O. First a `cupy.ndarray` must be initialized which is large enough to contain the requested byte range before making a read call. Then we pass the buffer to `pread` where the size of the read is inferred implicitly from `buf.size`. `kvikio.cufile.CuFile.pread()` returns an `IOFuture` which must be waited on with `IOFuture.get()` since `pread()` is non-blocking.

# A Word on Compatibility Modes
Both cuFile and kvikio provide similar but distinctly separate compatbility modes. It is not strictly required to make cuFile API calls in a GDS supported set-up. When GDS is not supported cuFile goes to compatibility mode and read/write operations are made using the POSIX API. 

Kvikio similarly provides a compatibility mode when it is unable to load `libcufile`. When kvikio is in compatibility mode, all read/write operations are made using the POSIX API. 

CuFile and Kvikio have separate compatibility modes. That is CuFile can be ran in compatibility mode when kvikio is not. KvikIO's compatibility mode is provided for performance considerations and will bypass the cuFile API to make the POSIX backend API calls directly.

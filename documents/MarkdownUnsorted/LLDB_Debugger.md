# LLDB Debugger

The [LLDB Debugger](https://lldb.llvm.org/) is part of the [LLVM project](https://llvm.org/) and is the default debugger on macOS supporting C, Objective-C and C++.

## LLVM project

The [LLVM project](https://github.com/llvm/llvm-project) (Low level virtual machine) is a collection of modular and reusable compiler and toolchain technologies and despite its name, LLVM has little to do with traditional virtual machines.

### Primary subprojects

* **LLVM Core** librarires provide a modern source- and target- independent optimier along with code generation support for many CPUs
* **Clang** is an *LLVM native* C/C++/Objective-C compiler
* **LLDB** is a native debugger
* **libc++** and **libc++ ABI** are hig-performance implementation of the C++ standard library
* **compiler-rt** provides highly tuned implementations of the low-level code generator support routines
* **MLIR** is a novel approach for building reusable and extensible compler infrastructure
* **OpenMP** provides an OpenMP runtime for use with the OpenMP implementation in Clang
* **polly** implements a suite of cache-locality optimizations as well as auto-parallelism and vectorization using a polyhedral model
* **libclc** aims to implement the OpenCL standard library
* **klee** implements a *symbolic virtual machine* using a theorem prover to try to evaluate all dynamic paths through a program in an effort to find bugs and to prove properties of functions
* **LLD** is a new linker, that is a drop-in replacement for system linkers (and runs much faster)

## Functionality

LLDB converts debug information into Clang types so that it can leverage the Clang compiler infrastructure. This allows LLDB to support the latest C, C++, Objective-C and Objective-C++ language features and runtimes in expressions without having to reimplement any of this functionality. It also leverages the compiler to take care of all ABI details when making functions calls for expressions, when disassembling instructions and extracting instruction details, and much more.

### Usage

* see the [GDB to LLDB](https://lldb.llvm.org/use/map.html) command map
* see a comprehensive [LLDB cheat sheet](https://www.nesono.com/sites/default/files/lldb%20cheat%20sheet.pdf)

#### Print out

* Print object

```
(lldb) po responseObject
(lldb) po [responseObject objectForKey@"state"]
```

* p - Print primitive type


### Breakpoints

* List breakpoints

```
br l
```

* br delete - Delete breakpoint

```
(lldb) br delete 1
```

* br e - Enable breakpoint
* br di - Disable breakpoint
* b - Add breakpoint

```
(lldb) b MyViewController.m:30
```

* br set - Add symbolic breakpoint

```
(lldb) br set -n viewDidLoad
```

* Conditional break

``` objc
for(PlayerItem *item in player.inventory) {
    totalValue += item.value;
}
```

Set a conditional breakpoint that triggers only when `totalValue` is greater than 1000:

```
(lldb) b MerchantViewController.m:32
Breakpoint 3: where = lootcollector`-[MerchantViewController] ...
(lldb) br mod -c "totalValue > 1000" 3
```

Reset the condition:

```
(lldb) br mod -c "" 3
```

* Run a debugger command from a breakpoint

```
(lldb) br com add 2
Enter your debugger command(s). Type 'DONE' to end.
> bt
> continue
> DONE
```

* Resume excution

```
(lldb) continue
```

* Step over

```
(lldb) n
```

* Step in

```
(lldb) s
```

* Print backtrace

```
(lldb) bt
```


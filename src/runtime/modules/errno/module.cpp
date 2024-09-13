#include "../Modules.hpp"
#include "runtime/PyDict.hpp"
#include <array>
#include <cerrno>

namespace py {

namespace {
	static constexpr std::string_view kDoc =
		R"(This module makes available standard errno system symbols.

The value of each symbol is the corresponding integer value,
e.g., on most systems, errno.ENOENT equals the integer 2.

The dictionary errno.errorcode maps numeric codes to symbol names,
e.g., errno.errorcode[2] could be the string 'ENOENT'.

Symbols that are not relevant to the underlying system are not defined.

To map error codes to error messages, use the function os.strerror(),
e.g. os.strerror(2) could return 'No such file or directory'.)";
}

// TODO: these errno values are Linux specific
static constexpr std::array kErrnoValues{
	std::tuple{ EPERM, "EPERM" },
	std::tuple{ ENOENT, "ENOENT" },
	std::tuple{ ESRCH, "ESRCH" },
	std::tuple{ EINTR, "EINTR" },
	std::tuple{ EIO, "EIO" },
	std::tuple{ ENXIO, "ENXIO" },
	std::tuple{ E2BIG, "E2BIG" },
	std::tuple{ ENOEXEC, "ENOEXEC" },
	std::tuple{ EBADF, "EBADF" },
	std::tuple{ ECHILD, "ECHILD" },
	std::tuple{ EAGAIN, "EAGAIN" },
	std::tuple{ ENOMEM, "ENOMEM" },
	std::tuple{ EACCES, "EACCES" },
	std::tuple{ EFAULT, "EFAULT" },
	std::tuple{ EBUSY, "EBUSY" },
	std::tuple{ EEXIST, "EEXIST" },
	std::tuple{ EXDEV, "EXDEV" },
	std::tuple{ ENODEV, "ENODEV" },
	std::tuple{ ENOTDIR, "ENOTDIR" },
	std::tuple{ EISDIR, "EISDIR" },
	std::tuple{ EINVAL, "EINVAL" },
	std::tuple{ ENFILE, "ENFILE" },
	std::tuple{ EMFILE, "EMFILE" },
	std::tuple{ ENOTTY, "ENOTTY" },
	std::tuple{ ETXTBSY, "ETXTBSY" },
	std::tuple{ EFBIG, "EFBIG" },
	std::tuple{ ENOSPC, "ENOSPC" },
	std::tuple{ ESPIPE, "ESPIPE" },
	std::tuple{ EROFS, "EROFS" },
	std::tuple{ EMLINK, "EMLINK" },
	std::tuple{ EPIPE, "EPIPE" },
	std::tuple{ EDOM, "EDOM" },
	std::tuple{ ERANGE, "ERANGE" },
	std::tuple{ EDEADLK, "EDEADLK" },
	std::tuple{ ENAMETOOLONG, "ENAMETOOLONG" },
	std::tuple{ ENOLCK, "ENOLCK" },
	std::tuple{ ENOSYS, "ENOSYS" },
	std::tuple{ ENOTEMPTY, "ENOTEMPTY" },
	std::tuple{ ELOOP, "ELOOP" },
	std::tuple{ EWOULDBLOCK, "EWOULDBLOCK" },
	std::tuple{ ENOMSG, "ENOMSG" },
	std::tuple{ EIDRM, "EIDRM" },
	std::tuple{ ENOSTR, "ENOSTR" },
	std::tuple{ ENODATA, "ENODATA" },
	std::tuple{ ETIME, "ETIME" },
	std::tuple{ ENOSR, "ENOSR" },
	std::tuple{ EREMOTE, "EREMOTE" },
	std::tuple{ ENOLINK, "ENOLINK" },
	std::tuple{ EPROTO, "EPROTO" },
	std::tuple{ EBADMSG, "EBADMSG" },
	std::tuple{ EOVERFLOW, "EOVERFLOW" },
	std::tuple{ EILSEQ, "EILSEQ" },
	std::tuple{ EUSERS, "EUSERS" },
	std::tuple{ ENOTSOCK, "ENOTSOCK" },
	std::tuple{ EDESTADDRREQ, "EDESTADDRREQ" },
	std::tuple{ EMSGSIZE, "EMSGSIZE" },
	std::tuple{ EPROTOTYPE, "EPROTOTYPE" },
	std::tuple{ ENOPROTOOPT, "ENOPROTOOPT" },
	std::tuple{ EPROTONOSUPPORT, "EPROTONOSUPPORT" },
	std::tuple{ ESOCKTNOSUPPORT, "ESOCKTNOSUPPORT" },
	std::tuple{ ENOTSUP, "ENOTSUP" },
	std::tuple{ EOPNOTSUPP, "EOPNOTSUPP" },
	std::tuple{ EPFNOSUPPORT, "EPFNOSUPPORT" },
	std::tuple{ EAFNOSUPPORT, "EAFNOSUPPORT" },
	std::tuple{ EADDRINUSE, "EADDRINUSE" },
	std::tuple{ EADDRNOTAVAIL, "EADDRNOTAVAIL" },
	std::tuple{ ENETDOWN, "ENETDOWN" },
	std::tuple{ ENETUNREACH, "ENETUNREACH" },
	std::tuple{ ENETRESET, "ENETRESET" },
	std::tuple{ ECONNABORTED, "ECONNABORTED" },
	std::tuple{ ECONNRESET, "ECONNRESET" },
	std::tuple{ ENOBUFS, "ENOBUFS" },
	std::tuple{ EISCONN, "EISCONN" },
	std::tuple{ ENOTCONN, "ENOTCONN" },
	std::tuple{ ESHUTDOWN, "ESHUTDOWN" },
	std::tuple{ ETOOMANYREFS, "ETOOMANYREFS" },
	std::tuple{ ETIMEDOUT, "ETIMEDOUT" },
	std::tuple{ ECONNREFUSED, "ECONNREFUSED" },
	std::tuple{ EHOSTDOWN, "EHOSTDOWN" },
	std::tuple{ EHOSTUNREACH, "EHOSTUNREACH" },
	std::tuple{ EALREADY, "EALREADY" },
	std::tuple{ EINPROGRESS, "EINPROGRESS" },
	std::tuple{ ESTALE, "ESTALE" },
	std::tuple{ EDQUOT, "EDQUOT" },
	std::tuple{ ECANCELED, "ECANCELED" },
	std::tuple{ ENOTRECOVERABLE, "ENOTRECOVERABLE" },
	std::tuple{ EOWNERDEAD, "EOWNERDEAD" },
	std::tuple{ EDEADLOCK, "EDEADLOCK" },
	std::tuple{ ECHRNG, "ECHRNG" },
	std::tuple{ EL2NSYNC, "EL2NSYNC" },
	std::tuple{ EL3HLT, "EL3HLT" },
	std::tuple{ EL3RST, "EL3RST" },
	std::tuple{ ELNRNG, "ELNRNG" },
	std::tuple{ EUNATCH, "EUNATCH" },
	std::tuple{ ENOCSI, "ENOCSI" },
	std::tuple{ EL2HLT, "EL2HLT" },
	std::tuple{ EBADE, "EBADE" },
	std::tuple{ EBADR, "EBADR" },
	std::tuple{ EXFULL, "EXFULL" },
	std::tuple{ ENOANO, "ENOANO" },
	std::tuple{ EBADRQC, "EBADRQC" },
	std::tuple{ EBADSLT, "EBADSLT" },
	std::tuple{ EBFONT, "EBFONT" },
	std::tuple{ ENONET, "ENONET" },
	std::tuple{ ENOPKG, "ENOPKG" },
	std::tuple{ EADV, "EADV" },
	std::tuple{ ESRMNT, "ESRMNT" },
	std::tuple{ ECOMM, "ECOMM" },
	std::tuple{ EDOTDOT, "EDOTDOT" },
	std::tuple{ ENOTUNIQ, "ENOTUNIQ" },
	std::tuple{ EBADFD, "EBADFD" },
	std::tuple{ EREMCHG, "EREMCHG" },
	std::tuple{ ELIBACC, "ELIBACC" },
	std::tuple{ ELIBBAD, "ELIBBAD" },
	std::tuple{ ELIBSCN, "ELIBSCN" },
	std::tuple{ ELIBMAX, "ELIBMAX" },
	std::tuple{ ELIBEXEC, "ELIBEXEC" },
	std::tuple{ ERESTART, "ERESTART" },
	std::tuple{ ESTRPIPE, "ESTRPIPE" },
	std::tuple{ EUCLEAN, "EUCLEAN" },
	std::tuple{ ENOTNAM, "ENOTNAM" },
	std::tuple{ ENAVAIL, "ENAVAIL" },
	std::tuple{ EISNAM, "EISNAM" },
	std::tuple{ EREMOTEIO, "EREMOTEIO" },
	std::tuple{ EKEYEXPIRED, "EKEYEXPIRED" },
	std::tuple{ EKEYREJECTED, "EKEYREJECTED" },
	std::tuple{ EKEYREVOKED, "EKEYREVOKED" },
	std::tuple{ EMEDIUMTYPE, "EMEDIUMTYPE" },
	std::tuple{ ENOKEY, "ENOKEY" },
	std::tuple{ ENOMEDIUM, "ENOMEDIUM" },
	std::tuple{ ERFKILL, "ERFKILL" },
};

PyModule *errno_module()
{
	auto symbols = PyDict::create().unwrap();
	auto name = PyString::create("errno").unwrap();
	auto doc = PyString::create(std::string{ kDoc }).unwrap();

	auto *module = PyModule::create(symbols, name, doc).unwrap();

	PyDict::MapType errorcode;
	for (auto [errno_value, errno_name] : kErrnoValues) {
		module->add_symbol(
			PyString::create(std::string{ errno_name }).unwrap(), Number{ errno_value });
		errorcode[Number{ errno_value }] = String{ errno_name };
	}
	module->add_symbol(PyString::create("errorcode").unwrap(), PyDict::create(errorcode).unwrap());

	return module;
}
}// namespace py

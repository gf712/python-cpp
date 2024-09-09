#include "../Modules.hpp"
#include "runtime/PyDict.hpp"
#include "runtime/PyType.hpp"

namespace py {

namespace {
	static constexpr std::string_view kDoc =
		R"(This module provides various functions to manipulate time values.

There are two standard representations of time.  One is the number
of seconds since the Epoch, in UTC (a.k.a. GMT).  It may be an integer
or a floating point number (to represent fractions of seconds).
The Epoch is system-defined; on Unix, it is generally January 1st, 1970.
The actual value can be retrieved by calling gmtime(0).

The other representation is a tuple of 9 integers giving local time.
The tuple items are:
  year (including century, e.g. 1998)
  month (1-12)
  day (1-31)
  hours (0-23)
  minutes (0-59)
  seconds (0-59)
  weekday (0-6, Monday is 0)
  Julian day (day in the year, 1-366)
  DST (Daylight Savings Time) flag (-1, 0 or 1)
If the DST flag is 0, the time is given in the regular time zone;
if it is 1, the time is given in the DST time zone;
if it is -1, mktime() should guess based on the date and time.)";
}

PyModule *time_module()
{
	auto symbols = PyDict::create().unwrap();
	auto name = PyString::create("time").unwrap();
	auto doc = PyString::create(std::string{ kDoc }).unwrap();

	auto *module = PyModule::create(symbols, name, doc).unwrap();

	return module;
}
}// namespace py

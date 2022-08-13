def reaches_finally_without_exception():
    reaches_finally = False
    try:
        print("try")
        a = 1
    except ValueError:
        a = 2
        print("RuntimeError")
    except TypeError:
        a = 3
        print("TypeError")
    finally:
        reaches_finally = True
        print("finally")

    assert a == 1, "Integer assignment should not raise an exception"
    assert reaches_finally, "Finally should always be reached"

reaches_finally_without_exception()

def get_exception():
    print("foo")
    return TypeError

def catches_exception_using_return_value_in_exception_handle():
    reaches_finally = False
    try:
        print("try")
        a = 1
        b = "a"
        c = a + b
    except ValueError:
        a = 2
        print("ValueError")
    except get_exception():
        a = 3
        print("TypeError")
    finally:
        reaches_finally = True
        print("finally")

    assert a == 3, "Integer and string addition should result in a TypeError"
    assert reaches_finally, "Finally should always be reached"

catches_exception_using_return_value_in_exception_handle()

def try_catches_using_base_exception_in_handle():
    reaches_finally = False
    try:
        print("try")
        a = 1
        b = "a"
        c = a + b
    except ValueError:
        a = 2
        print("ValueError")
    except Exception:
        a = 3
        print("TypeError")
    finally:
        reaches_finally = True
        print("finally")

    assert a == 3, "Integer and string addition should result in an exception that is derived from Exception"
    assert reaches_finally, "Finally should always be reached"

try_catches_using_base_exception_in_handle()

def try_catches_all_exceptions_with_empty_handle():
    reaches_finally = False
    try:
        a = 1
        b = "a"
        c = a + b
    except ValueError:
        a = 2
    except:
        a = 3
    finally:
        reaches_finally = True

    assert a == 3, "Default exception should catch all exceptions"
    assert reaches_finally, "Finally should always be reached"

try_catches_all_exceptions_with_empty_handle()

def nested_try_with_inner_except_return():
    try:
        try:
            a = 1 + "foo"
            return 1
        except:
            return 2
        finally:
            print("bar1")
    except:
        print("bar2")
        return 10

assert nested_try_with_inner_except_return() == 2

def nested_try_with_outer_except_return():
    try:
        try:
            a = 1 + "foo"
            return 1
        finally:
            print("bar1")
    except:
        print("bar2")
        return 10

assert nested_try_with_outer_except_return() == 10

def try_else_return_from_exception():
    try:
        a = 1 + "foo "
    except:
        return 2
    else:
        return 3
    finally:
        pass

assert try_else_return_from_exception() == 2

def try_else():
    try:
        a = 1
    except:
        a = 2
    else:
        a = 3
    finally:
        a = 4
    return a

assert try_else() == 4

def try_else_return_from_else():
    try:
        a = 1
    except:
        return 2
    else:
        return 3
    finally:
        pass

assert try_else_return_from_else() == 3

def try_else_return_from_finally():
    try:
        a = 1
    except:
        return 2
    else:
        return 3
    finally:
        return 4

assert try_else_return_from_finally() == 4

def try_else_return_from_finally_with_exception():
    try:
        a = 1 + "foo"
    except:
        return 2
    else:
        return 3
    finally:
        return 4

assert try_else_return_from_finally() == 4
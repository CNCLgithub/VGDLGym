using Gen


@gen (static) function foo(x::Float64, y::Float64)
    a ~ normal(x, 1.0)
    b ~ normal(y, abs(a))
    result::Float64 = a + b
    return result
end


@gen (static) function bar(x::Float64, y::Float64)
    z ~ normal(x, y)
    return z
end

foobar = Gen.Switch(foo, bar)

@gen (static) function buzz()
    x ~ uniform(0., 1.0)
    y ~ uniform(5.0, 10.0)
    c ~ categorical([0.5, 0.5])
    sw ~ foobar(c, x, y)
    k ~ normal(sw, 1.0)
    return k
end

function test()
    tr, ls = Gen.generate(buzz, (), choicemap(:c => 1))
    @show ls
    display(get_choices(tr))


    tr, ls, rdiff, discard = Gen.update(tr, (), (), choicemap(:c => 2))
    @show ls
    display(get_choices(tr))
    @show rdiff
    display(discard)
end


test();

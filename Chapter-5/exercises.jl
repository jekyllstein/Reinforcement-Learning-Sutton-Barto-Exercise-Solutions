function sieve(plist, primes = Vector{Int64}())
    isempty(plist) && return primes
    p = first(plist)
    newlist = plist |> Drop(1) |> Filter(x -> x % p != 0)
    sieve(newlist, push!(primes, p))
end

function makenewlist(plist)
    p = first(plist)
    newlist = plist |> Drop(1) |> Filter(x -> x % p != 0)
    return (p, newlist)
end

primes = countfrom(2) |> ScanEmit((u, x) -> makenewlist(u), countfrom(2))

struct PrimeIter end

Base.iterate(::PrimeIter) = (2, countfrom(3, 2))
function Base.iterate(::PrimeIter, state)
    n = first(state)
    nextstate = Iterators.filter(i -> i%n != 0, state)
    (n, nextstate)
end

primes = PrimeIter()

function insortediter(item, iter)
    out = false
    for i in iter
        if i == item
            out = true
            break
        elseif i > item
            break
        end
    end
    return out
end 

#this produces stack overflows without transducers and with tranducers is very slow
struct PrimeIterEuler end
Base.iterate(::PrimeIterEuler) = (2, countfrom(3,2))
function Base.iterate(::PrimeIterEuler, state)
    n = first(state)
    #these values are to be removed by the filter
    filtlist = state |> Map(x -> x*n)
    nextstate = state |> Drop(1) |> Filter(i -> !insortediter(i, filtlist))
    (n, nextstate)
end


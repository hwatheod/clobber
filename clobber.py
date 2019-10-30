# Python 2 only
"""
This code calculates atomic weights for near-alternating Clobber positions.  A near-alternating Clobber position
has the form X...Y where X and Y are one of B,W,BB,WW, and ... denotes an alternating sequence of black and white
stones that begins with the opposite color of X and ends with the opposite color of Y.

In the code below, such a Clobber position is represented as (left_end, n, right_end) where left_end is X, right_end is Y,
and n is the number of stones in the "..." sequence.  left_end and right_end are represented as strings of b's and w's.

We also allow n = -1 and n = -2 because that simplifies the formulas for options:
  (BB, -2, WW) = BW
  (WW, -2, BB) = WB
  In all other cases, (X, -1, Y) and (X, -2, Y) denote the empty position.

The following abbreviations for references are used in some comments:
 LiP: "Lessons in Play: An Introduction to Combinatorial Game Theory" by Albert, Nowakowski, and Wolfe.
 WW: "Winning Ways for Your Mathematical Plays" by Berlekamp, Conway, and Guy.
"""
from string import maketrans
import math

# negate either a number or '*'
def neg(s):
    if type(s) == float or type(s) == int:
        return -s
    if s == '*':
        return '*'
    raise ValueError('Unknown value passed to neg ' + str(s))

# negate a Clobber position (a string of b's and w's)
def invert(s):
    return s.translate(maketrans('bw','wb'))

# determine the parity of the expected number of stones in between the given left and right end
# returns 1 for odd and 0 for even.
def expected_parity(left_end, right_end):
    if left_end[0] == right_end[0]:
        return 1
    else:
        return 0

# cached values of atomic weights.  Some pre-calculated values are hardcoded here (calculated by cgsuite)
# because they require comparison with remote stars which is not implemented in this code.
values = {}
values[('b',0,'w')] = 0
values[('b',2,'w')] = 0
values[('b',4,'w')] = 0
values[('b',6,'w')] = 0

values[('bb',0,'ww')] = 0
values[('bb',2,'ww')] = 0
values[('bb',4,'ww')] = 0
values[('bb',6,'ww')] = '*'

values[('b',1,'b')] = 0
values[('b',3,'b')] = -1
values[('b',5,'b')] = -1

values[('bb',1,'bb')] = 1
values[('bb',3,'bb')] = 0
values[('bb',5,'bb')] = 1./2

values[('bb',0,'w')] = 1
values[('bb',2,'w')] = 0
values[('bb',4,'w')] = 1./2
values[('bb',6,'w')] = 1

values[('bb',1,'b')] = -1
values[('bb',3,'b')] = 0
values[('bb',5,'b')] = -1./2


def validate_parameters(left_end, n, right_end):
    # check input data
    if len(left_end) == 0 or len(right_end) == 0:
        raise ValueError('Empty argument for left_end or right_end is not valid.')
    ep = expected_parity(left_end, right_end)
    if n % 2 != ep:
        raise ValueError(str(n) + ' does not have expected parity ' + str(ep))
    if left_end != left_end[0] * len(left_end):
        raise ValueError('left_end value ' + left_end + ' is invalid, must be all black or all white')
    if right_end != right_end[0] * len(right_end):
        raise ValueError('right_end value ' + right_end + ' is invalid, must be all black or all white')
    if type(n) != int or n < -2:
        raise ValueError('value of n ' + str(n) + ' invalid.  Must be an integer >= -2')

# Get the atomic weight of the Clobber position (left_end, n, right_end).
# Applies symmetries, checks cache, and then delegates to calculate_atomic_weight.
def atomic_weight(left_end, n, right_end):
    global values

    validate_parameters(left_end, n, right_end)
    # boundary cases 0,-1,-2
    if n == 0:
        if len(left_end) >= 2 and len(right_end) >= 2: # B^n W^m has value 0 if n,m > 1 (LiP, Exercise 9.49)
            return 0

        # Otherwise, W(B^n) has atomic weight n-1 (LiP, Lemma 9.48)
        if len(left_end) == 1:
            other_end = right_end
        else:
            other_end = left_end

        if other_end[0] == 'b':
            return len(other_end) - 1
        else:
            return 1 - len(other_end)
    if n == -1 or n == -2:
        if len(left_end) > 2 and len(right_end) > 2:
            raise ValueError('Negative argument for n not valid unless at least one end has length <= 2.  Got: ' + str(left_end) + ' ' + str(right_end))
        return 0  # this is either BW or one solid color, atomic weight 0.

    # Take symmetries into account
    if len(left_end) < len(right_end):
        return atomic_weight(right_end, n, left_end)
    if left_end[0] == 'w':
        return neg(atomic_weight(invert(left_end), n, invert(right_end)))

    # Now left_end is black and len(left_end) >= len(right_end)
    # We treat n=1 specially because it requires remote star comparisons otherwise.
    # See LiP, Lemma 9.52.
    if n == 1: 
        if len(left_end) == 2 and len(right_end) == 1:  # "split path" BBWB
            return -1
        else: # remember we symmetrized so that the right_end is shorter
            return len(right_end) - 1
        
    # look up value if we have it cached already
    if (left_end, n, right_end) in values:
        return values[(left_end, n, right_end)]

    return calculate_atomic_weight(left_end, n, right_end)


# We only handle atomic weight values of the form number and number plus star.
#   number is represented as a python numeric value.
#   star is represented as the string '*'.
#   number plus star is represented as the tuple (number, '*').
# We call this the "standard form".

# Write the standard form value aw in the form (number, boolean) where if the boolean is True we add a * to the number,
# and if False, we don't. We call this "component form".
def component(aw):
    if aw == '*':
        return (0, True)
    if type(aw) == int or type(aw) == float:
        return (aw, False)
    if type(aw) == tuple and (type(aw[0]) == int or type(aw[0]) == float) and aw[1] == '*':
        return (aw[0], True)
    raise ValueError('Unknown value ' + str(aw))

# Normalize an atomic weight value in component form back to standard form.
def normalize(comp):
    if comp[1]: # has a star
        if comp[0] != 0:
            return (comp[0], '*')
        else:
            return '*'
    else:
        return comp[0]

# Add 2 standard form values and return the value in standard form.
def awsum(aw1, aw2):
    [comp1, comp2] = [component(aw1), component(aw2)]
    s = (comp1[0] + comp2[0], comp1[1] ^ comp2[1])
    return normalize(s)

# Compare 2 standard form values.  Return:
#   1 if aw1 > aw2
#   0 if aw1 = aw2
#  -1 if aw1 < aw2
#   | if aw1 || aw2 (confused)
def gt(aw1, aw2):
    awcomps = map(component, [aw1, aw2])
    if awcomps[0][1] == awcomps[1][1] or awcomps[0][0] != awcomps[1][0]:
        return 1 if awcomps[0][0] > awcomps[1][0] else -1 if awcomps[0][0] < awcomps[1][0] else 0
    else: # at this point we have values that differ by *
        return '|'


# Return the atomic weights of the options of (left_end, n, right_end).
def aw_of_options(left_end, n, right_end):
    validate_parameters(left_end, n, right_end)

    aw_left_options = []
    aw_right_options = []
    for r in xrange(-2, n - 2 + 1):
        s = n - 4 - r
        if r % 2 == expected_parity(left_end, 'bb'):
            aw_left_options.append(awsum(atomic_weight(left_end, r, 'bb'), atomic_weight('w', s, right_end)))

        if r % 2 == expected_parity(left_end, 'w'):
            aw_left_options.append(awsum(atomic_weight(left_end, r, 'w'), atomic_weight('bb', s, right_end)))

        if r % 2 == expected_parity(left_end, 'ww'):
            aw_right_options.append(awsum(atomic_weight(left_end, r, 'ww'), atomic_weight('b', s, right_end)))

        if r % 2 == expected_parity(left_end, 'b'):
            aw_right_options.append(awsum(atomic_weight(left_end, r, 'b'), atomic_weight('ww', s, right_end)))

    return (aw_left_options, aw_right_options)

# Calculate the atomic weight of the Clobber position (left_end, n, right_end).
def calculate_atomic_weight(left_end, n, right_end):
    validate_parameters(left_end, n, right_end)

    # find the maximal atomic weight of left options, and minimal atomic weight of right options.
    max_left = []
    min_right = []
    aw_left_options, aw_right_options = aw_of_options(left_end, n, right_end)
    for aw_left_option in aw_left_options:
        new_maximal = True
        for i,v in enumerate(max_left):
            comp = gt(aw_left_option, v)
            if comp == 0 or comp == -1:
                new_maximal = False
                break
            if comp == 1:
                max_left[i] = None  # this left option is no longer maximal, we will delete it after this loop.
        if new_maximal:
            while max_left.count(None) > 0:
                max_left.remove(None)
            max_left.append(aw_left_option)
    for aw_right_option in aw_right_options:
        new_minimal = True
        for i,v in enumerate(min_right):
            comp = gt(aw_right_option, v)
            if comp == 0 or comp == 1:
                new_minimal = False
                break
            if comp == -1:
                min_right[i] = None # this right option is no longer minimal, we will delete it after this loop.
        if new_minimal:
            while min_right.count(None) > 0:
                min_right.remove(None)
            min_right.append(aw_right_option)

    # per the atomic weight calculus, subtract 2 from left atomic weights and add 2 to right atomic weights.
    adjusted_aw_left_options, adjusted_aw_right_options = \
        (map(lambda x: awsum(x, -2), max_left), map(lambda x: awsum(x, 2), min_right))
    if len(adjusted_aw_left_options) <= 1 and len(adjusted_aw_right_options) <= 1:
        if len(adjusted_aw_left_options) == 0 and len(adjusted_aw_right_options) == 0:
            # neither player has any options, zero game
            return 0
        if len(adjusted_aw_left_options) == 0 or len(adjusted_aw_right_options) == 0:
            # cannot have one side empty and other side nonempty, since Clobber is a dicotic game.
            raise ValueError('Unexpected atomic weight calculation, one side is empty and the other is not')
        try:
            aw = simplestform((adjusted_aw_left_options[0], adjusted_aw_right_options[0]))
            values[(left_end,n,right_end)] = aw
            return aw
        except ValueError as msg:
            raise ValueError('Unable to calculate simplest form at (' + left_end + ',' + str(n) + ',' + right_end + ').  Message: ' + str(msg))
    else:
        raise ValueError('Unable to calculate simplest form at (' + left_end + ',' + str(n) + ',' + right_end + ') because there are 2 or more undominated options on one side.')

# Attempt to calculate the simplest form of a game of the form {L|R} where L and R are one of the supported "standard form"
# atomic weights (see above). Fail if a comparison with a remote star is needed per the atomic weight calculus.
#
# Not all games are supported, just enough to handle the cases that arise in the calculation of near-alternating
# Clobber positions.  A ValueError is raised for an unsupported case.

def simplestform(game):
    (left, right) = game
    leftcomp = component(left)
    rightcomp = component(right)
    if leftcomp[0] > rightcomp[0]:  # hot game, not expected to happen
        raise ValueError('Unexpected hot game ' + str(game))
    if leftcomp[0] == rightcomp[0]:
        if leftcomp[1] == rightcomp[1]:  # like {3|3} or {3*|3*}
            return normalize((leftcomp[0], not leftcomp[1]))
        else: # something like {3|3*}, also not expected to happen
            raise ValueError('Unexpected game ' + str(game))
    if leftcomp[0] < rightcomp[0]:
        if int(leftcomp[0]) == leftcomp[0]:
            if leftcomp[1]: # left option is integer plus star
                integer_above_left = int(leftcomp[0])
            else: # left option is integer
                integer_above_left = int(leftcomp[0] + 1)
        else: # left option is non-integer
            integer_above_left = int(math.ceil(leftcomp[0]))

        if int(rightcomp[0]) == rightcomp[0]:
            if rightcomp[1]: # right option is integer plus star
                integer_below_right = int(rightcomp[0])
            else: # right option is integer
                integer_below_right = int(rightcomp[0] - 1)
        else: # right option is non-integer
            integer_below_right = int(math.floor(rightcomp[0]))

        if integer_above_left == integer_below_right:
            # only one integer fits
            # no remote star comparison needed (WW, 2nd ed., Volume 1, Chapter 8, p. 251, "Proper Care of the Eccentric")
            return integer_below_right
        if integer_above_left < integer_below_right: # two or more integers fit
            raise ValueError('Remote star comparison needed ' + str(game))

        # no integers fit, yet numerical part of left < numerical part of right.
        # Left and right values must be in [n, n+1] for some integer n.
        assert integer_above_left - integer_below_right == 1, 'Unexpected values ' + str(game) + ' above, below values are: ' + str(integer_above_left) + ' ' + str(integer_below_right)

        integer_part = integer_below_right
        l = leftcomp[0] - integer_below_right
        r = rightcomp[0] - integer_below_right
        mid = 0.5
        while not (l < mid < r):
            if l < mid:
                r = mid
            elif mid < r:
                l = mid
            else:
                raise ValueError('Shouldn\'t get here ' + str(game) + ' above, below values are: ' + str(integer_above_left) + ' ' + str(integer_below_right) + ' l r mid ' + str(l) + ' ' + str(r) + ' ' + str(mid))
            mid = (l + r) / 2.
        return integer_part + mid
    raise ValueError('Unable to get simplest form ' + str(game))

# Evaluate atomic weights for all near-alternating Clobber positions with a <= n <= b.
def evalall(a,b):
    for c in xrange(a,b+1):
        if c % 2 == 0:
            atomic_weight('b', c, 'w')
            atomic_weight('bb', c, 'ww')
            atomic_weight('bb', c, 'w')
        else:
            atomic_weight('b', c, 'b')
            atomic_weight('bb', c, 'bb')
            atomic_weight('bb', c, 'b')

# Evaluate and print atomic weights for all near-alternating Clobber positions with a <= n <= b.
def printresults(a,b):
    print ('.\tb-w\tbb-ww\tb-b\tbb-bb\tbb-w\tbb-b')
    for c in xrange(a, b+1):
        res = ['.']*6
        if c % 2 == 0:            
            res[0] = atomic_weight('b', c, 'w')
            res[1] = atomic_weight('bb', c, 'ww')
            res[4] = atomic_weight('bb', c, 'w')
        else:
            res[2] = atomic_weight('b', c, 'b')
            res[3] = atomic_weight('bb', c, 'bb')
            res[5] = atomic_weight('bb', c, 'b')
        print (str(c) + '\t' + '\t'.join(map(str,res)))

#printresults(0, 30)
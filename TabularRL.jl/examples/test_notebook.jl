### A Pluto.jl notebook ###
# v0.19.42

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 2e75e7a8-66e7-4228-a85f-6b32ba933018
# ╠═╡ show_logs = false
begin
	using Pkg
	Pkg.activate(mktempdir())
	Pkg.develop(path = joinpath(@__DIR__, "..", "..", "TabularRL.jl"))
	Pkg.add(["StatsBase", "DataFrames", "Statistics", "DataStructures", "StaticArrays", "Transducers", "Serialization", "PlutoHooks", "LinearAlgebra", "SparseArrays", "HypertextLiteral", "Markdown"])
	using TabularRL, Statistics, StatsBase, DataFrames, DataStructures, StaticArrays, Transducers, Serialization, PlutoHooks, LinearAlgebra, SparseArrays, HypertextLiteral, Markdown

	@skip_as_script html"""
	<style>
		main {
			margin: 0 auto;
			max-width: min(1200px, 90%);
	    	padding-left: max(10px, 5%);
	    	padding-right: max(10px, 5%);
			font-size: max(10px, min(24px, 2vw));
		}
	</style>
	"""
end

# ╔═╡ 3553570d-970e-4e1d-929b-19387e79a31e
# ╠═╡ show_logs = false
Pkg.add("BenchmarkTools")

# ╔═╡ 4225d99f-c30f-47e8-b6c1-9a167d4e937c
# ╠═╡ show_logs = false
@skip_as_script begin
	Pkg.add(["PlutoUI", "PlutoPlotly", "PlutoProfile", "AbstractPlutoDingetjes"])
	using PlutoUI, PlutoPlotly, PlutoProfile
	import AbstractPlutoDingetjes.Bonds
	TableOfContents()
end

# ╔═╡ 464a4be1-1fa2-4d60-9fb8-72fc47723cf5
using BenchmarkTools

# ╔═╡ df9a4dd8-5a36-497b-a61f-d76a830f4398
using Base.Threads

# ╔═╡ 7be1e611-0cba-4bf8-876d-757dd3931016
md"""
# TabularRL Functionality Tests

Verify that the TabularRL package is successfully loaded into this notebook and environment is suitable for creating new environments
"""

# ╔═╡ aa5ce283-909c-4752-8e1b-71b09720ae1b
value_iteration_v(make_deterministic_gridworld().mdp, 0.9f0)

# ╔═╡ f9dc6d6f-e417-482c-842b-c3d8ddaedd0e
value_iteration_v(make_stochastic_gridworld().mdp, 0.9f0)

# ╔═╡ 5d564193-c55b-4584-96ff-5b1cf404e334
md"""
# Test Environments
"""

# ╔═╡ b04c7371-c26a-4400-8dae-06b922b27af1
md"""
## Wordle Environment
"""

# ╔═╡ 589fac35-d61b-4ece-a316-610b91f26640
md"""
### Game Description

Wordle is a game of guessing 5 letter words from a predefined pool of allowed guesses.  A player has six attempts to guess the correct word and the game ends after a correct guess or when six incorrect guesses have been made.  After each guess, the player receives feedback per letter according to the following rules:

- Green: A letter is colored green if it matches the position in the answer
- Yellow: A letter is colored yellow if it is present in the answer but in a different location.  If the letter appears once in the answer but multiple times in the guess, only the first instance of the letter will be yellow.  If a letter appears more than once in the answer, then it could appear yellow multiple times in the guess as well.  This feedback is the most nuanced given its behavior changes depending on previous letters including if any were marked green
- Gray: A letter is marked gray if it does not appear in the answer

Throughout the game, a player can see the feedback for each prior guess.  A correct guess will receive the unique feedback of five green letters.
"""

# ╔═╡ 959f4088-9a24-4104-ad2e-1d1a8edfc3b2
md"""
### Letters and Feedback

Before creating the MDP, it is important to save precomputed values and structures to make step evaluations easier
"""

# ╔═╡ 502552da-745c-4991-a133-6f786191b255
begin
	#three different types of feedback corresponding to green, yellow and gray
	const EXACT = 0x02
	const MISPLACED = 0x01
	const MISSING = 0x00
	const letters = collect('a':'z')
	const letterlookup = Dict(zip(letters, UInt8.(1:length(letters))))
end

# ╔═╡ cf9cf9d8-b194-4b1d-afcd-22229ab0891b
md"""
### Example Feedback and Display
"""

# ╔═╡ 6851d75d-075c-47f8-8da6-f207e1382ccf
const example_feedback = [MISSING, EXACT, MISPLACED, MISPLACED, MISSING]

# ╔═╡ 6d771e10-781c-49bd-bf66-417a751713eb
md"""
### Wordle Feedback Visualization Example
"""

# ╔═╡ 451219d2-c0c6-4a28-ad27-9459cf860aa1
md"""
#### Observed feedback in game
"""

# ╔═╡ 56f7a7a1-721b-44e8-8cbb-253de9ab3d3d
md"""
#### Numerical feedback values
"""

# ╔═╡ 026063c3-705a-4d58-b3f6-0563485afe32
"""
	convert_bytes(v::AbstractVector{T}) where T <: Integer -> Integer

Convert a vector of ternary values to an integer.

Arguments

- v: A vector of integers representing ternary values i.e. 0, 1, 2

Returns

- An integer representing the equivalent value of the number in base 10
"""
convert_bytes(v::AbstractVector{T}) where T <: Integer = enumerate(v) |> Map(a -> a[2]*3^(a[1]-1)) |> sum

# ╔═╡ f38a4cf6-9daf-48ad-83f4-36e95020d13f
"""
	make_word_vec(word::AbstractString) -> SVector{5, UInt8}

Convert a word to a vector of byte representations.

Arguments

- word: A 5-character string

Returns

- A 5-element vector of bytes, where each element is the byte representation of a letter in the word

Details

This function uses the letterlookup table to map each letter in the word to its corresponding byte value.

See Also

- letterlookup
"""
make_word_vec(word::AbstractString) = SVector{5, UInt8}(letterlookup[word[i]] for i in 1:5)

# ╔═╡ d7544672-091a-4438-9b08-4d49b781dccb
#Read feedback bytes into a matrix of the proper size.  Must know the number of words in the original computation
function read_feedback(fname, l)
	output = Matrix{UInt8}(undef, l, l)
	f = open(fname)
	read!(f, output)
	close(f)
	return output
end

# ╔═╡ 75f63d9b-6d79-4113-8ad7-cb3833d02c23
md"""
###  Word Data
"""

# ╔═╡ 2b3ba2f2-bc89-4226-b60e-c05306f5886b
md"""
The following words were the original answers for the game which were predefined for each day.  Since the game was acquired by the New York Times, there is no longer a predefined answer list.
"""

# ╔═╡ 54347f08-3bc8-474d-9a1c-1888c4c126ea
const wordle_original_answers_raw = """aback
abase
abate
abbey
abbot
abhor
abide
abled
abode
abort
about
above
abuse
abyss
acorn
acrid
actor
acute
adage
adapt
adept
admin
admit
adobe
adopt
adore
adorn
adult
affix
afire
afoot
afoul
after
again
agape
agate
agent
agile
aging
aglow
agony
agree
ahead
aider
aisle
alarm
album
alert
algae
alibi
alien
align
alike
alive
allay
alley
allot
allow
alloy
aloft
alone
along
aloof
aloud
alpha
altar
alter
amass
amaze
amber
amble
amend
amiss
amity
among
ample
amply
amuse
angel
anger
angle
angry
angst
anime
ankle
annex
annoy
annul
anode
antic
anvil
aorta
apart
aphid
aping
apnea
apple
apply
apron
aptly
arbor
ardor
arena
argue
arise
armor
aroma
arose
array
arrow
arson
artsy
ascot
ashen
aside
askew
assay
asset
atoll
atone
attic
audio
audit
augur
aunty
avail
avert
avian
avoid
await
awake
award
aware
awash
awful
awoke
axial
axiom
axion
azure
bacon
badge
badly
bagel
baggy
baker
baler
balmy
banal
banjo
barge
baron
basal
basic
basil
basin
basis
baste
batch
bathe
baton
batty
bawdy
bayou
beach
beady
beard
beast
beech
beefy
befit
began
begat
beget
begin
begun
being
belch
belie
belle
belly
below
bench
beret
berry
berth
beset
betel
bevel
bezel
bible
bicep
biddy
bigot
bilge
billy
binge
bingo
biome
birch
birth
bison
bitty
black
blade
blame
bland
blank
blare
blast
blaze
bleak
bleat
bleed
bleep
blend
bless
blimp
blind
blink
bliss
blitz
bloat
block
bloke
blond
blood
bloom
blown
bluer
bluff
blunt
blurb
blurt
blush
board
boast
bobby
boney
bongo
bonus
booby
boost
booth
booty
booze
boozy
borax
borne
bosom
bossy
botch
bough
boule
bound
bowel
boxer
brace
braid
brain
brake
brand
brash
brass
brave
bravo
brawl
brawn
bread
break
breed
briar
bribe
brick
bride
brief
brine
bring
brink
briny
brisk
broad
broil
broke
brood
brook
broom
broth
brown
brunt
brush
brute
buddy
budge
buggy
bugle
build
built
bulge
bulky
bully
bunch
bunny
burly
burnt
burst
bused
bushy
butch
butte
buxom
buyer
bylaw
cabal
cabby
cabin
cable
cacao
cache
cacti
caddy
cadet
cagey
cairn
camel
cameo
canal
candy
canny
canoe
canon
caper
caput
carat
cargo
carol
carry
carve
caste
catch
cater
catty
caulk
cause
cavil
cease
cedar
cello
chafe
chaff
chain
chair
chalk
champ
chant
chaos
chard
charm
chart
chase
chasm
cheap
cheat
check
cheek
cheer
chess
chest
chick
chide
chief
child
chili
chill
chime
china
chirp
chock
choir
choke
chord
chore
chose
chuck
chump
chunk
churn
chute
cider
cigar
cinch
circa
civic
civil
clack
claim
clamp
clang
clank
clash
clasp
class
clean
clear
cleat
cleft
clerk
click
cliff
climb
cling
clink
cloak
clock
clone
close
cloth
cloud
clout
clove
clown
cluck
clued
clump
clung
coach
coast
cobra
cocoa
colon
color
comet
comfy
comic
comma
conch
condo
conic
copse
coral
corer
corny
couch
cough
could
count
coupe
court
coven
cover
covet
covey
cower
coyly
crack
craft
cramp
crane
crank
crash
crass
crate
crave
crawl
craze
crazy
creak
cream
credo
creed
creek
creep
creme
crepe
crept
cress
crest
crick
cried
crier
crime
crimp
crisp
croak
crock
crone
crony
crook
cross
croup
crowd
crown
crude
cruel
crumb
crump
crush
crust
crypt
cubic
cumin
curio
curly
curry
curse
curve
curvy
cutie
cyber
cycle
cynic
daddy
daily
dairy
daisy
dally
dance
dandy
datum
daunt
dealt
death
debar
debit
debug
debut
decal
decay
decor
decoy
decry
defer
deign
deity
delay
delta
delve
demon
demur
denim
dense
depot
depth
derby
deter
detox
deuce
devil
diary
dicey
digit
dilly
dimly
diner
dingo
dingy
diode
dirge
dirty
disco
ditch
ditto
ditty
diver
dizzy
dodge
dodgy
dogma
doing
dolly
donor
donut
dopey
doubt
dough
dowdy
dowel
downy
dowry
dozen
draft
drain
drake
drama
drank
drape
drawl
drawn
dread
dream
dress
dried
drier
drift
drill
drink
drive
droit
droll
drone
drool
droop
dross
drove
drown
druid
drunk
dryer
dryly
duchy
dully
dummy
dumpy
dunce
dusky
dusty
dutch
duvet
dwarf
dwell
dwelt
dying
eager
eagle
early
earth
easel
eaten
eater
ebony
eclat
edict
edify
eerie
egret
eight
eject
eking
elate
elbow
elder
elect
elegy
elfin
elide
elite
elope
elude
email
embed
ember
emcee
empty
enact
endow
enema
enemy
enjoy
ennui
ensue
enter
entry
envoy
epoch
epoxy
equal
equip
erase
erect
erode
error
erupt
essay
ester
ether
ethic
ethos
etude
evade
event
every
evict
evoke
exact
exalt
excel
exert
exile
exist
expel
extol
extra
exult
eying
fable
facet
faint
fairy
faith
false
fancy
fanny
farce
fatal
fatty
fault
fauna
favor
feast
fecal
feign
fella
felon
femme
femur
fence
feral
ferry
fetal
fetch
fetid
fetus
fever
fewer
fiber
ficus
field
fiend
fiery
fifth
fifty
fight
filer
filet
filly
filmy
filth
final
finch
finer
first
fishy
fixer
fizzy
fjord
flack
flail
flair
flake
flaky
flame
flank
flare
flash
flask
fleck
fleet
flesh
flick
flier
fling
flint
flirt
float
flock
flood
floor
flora
floss
flour
flout
flown
fluff
fluid
fluke
flume
flung
flunk
flush
flute
flyer
foamy
focal
focus
foggy
foist
folio
folly
foray
force
forge
forgo
forte
forth
forty
forum
found
foyer
frail
frame
frank
fraud
freak
freed
freer
fresh
friar
fried
frill
frisk
fritz
frock
frond
front
frost
froth
frown
froze
fruit
fudge
fugue
fully
fungi
funky
funny
furor
furry
fussy
fuzzy
gaffe
gaily
gamer
gamma
gamut
gassy
gaudy
gauge
gaunt
gauze
gavel
gawky
gayer
gayly
gazer
gecko
geeky
geese
genie
genre
ghost
ghoul
giant
giddy
gipsy
girly
girth
given
giver
glade
gland
glare
glass
glaze
gleam
glean
glide
glint
gloat
globe
gloom
glory
gloss
glove
glyph
gnash
gnome
godly
going
golem
golly
gonad
goner
goody
gooey
goofy
goose
gorge
gouge
gourd
grace
grade
graft
grail
grain
grand
grant
grape
graph
grasp
grass
grate
grave
gravy
graze
great
greed
green
greet
grief
grill
grime
grimy
grind
gripe
groan
groin
groom
grope
gross
group
grout
grove
growl
grown
gruel
gruff
grunt
guard
guava
guess
guest
guide
guild
guile
guilt
guise
gulch
gully
gumbo
gummy
guppy
gusto
gusty
gypsy
habit
hairy
halve
handy
happy
hardy
harem
harpy
harry
harsh
haste
hasty
hatch
hater
haunt
haute
haven
havoc
hazel
heady
heard
heart
heath
heave
heavy
hedge
hefty
heist
helix
hello
hence
heron
hilly
hinge
hippo
hippy
hitch
hoard
hobby
hoist
holly
homer
honey
honor
horde
horny
horse
hotel
hotly
hound
house
hovel
hover
howdy
human
humid
humor
humph
humus
hunch
hunky
hurry
husky
hussy
hutch
hydro
hyena
hymen
hyper
icily
icing
ideal
idiom
idiot
idler
idyll
igloo
iliac
image
imbue
impel
imply
inane
inbox
incur
index
inept
inert
infer
ingot
inlay
inlet
inner
input
inter
intro
ionic
irate
irony
islet
issue
itchy
ivory
jaunt
jazzy
jelly
jerky
jetty
jewel
jiffy
joint
joist
joker
jolly
joust
judge
juice
juicy
jumbo
jumpy
junta
junto
juror
kappa
karma
kayak
kebab
khaki
kinky
kiosk
kitty
knack
knave
knead
kneed
kneel
knelt
knife
knock
knoll
known
koala
krill
label
labor
laden
ladle
lager
lance
lanky
lapel
lapse
large
larva
lasso
latch
later
lathe
latte
laugh
layer
leach
leafy
leaky
leant
leapt
learn
lease
leash
least
leave
ledge
leech
leery
lefty
legal
leggy
lemon
lemur
leper
level
lever
libel
liege
light
liken
lilac
limbo
limit
linen
liner
lingo
lipid
lithe
liver
livid
llama
loamy
loath
lobby
local
locus
lodge
lofty
logic
login
loopy
loose
lorry
loser
louse
lousy
lover
lower
lowly
loyal
lucid
lucky
lumen
lumpy
lunar
lunch
lunge
lupus
lurch
lurid
lusty
lying
lymph
lyric
macaw
macho
macro
madam
madly
mafia
magic
magma
maize
major
maker
mambo
mamma
mammy
manga
mange
mango
mangy
mania
manic
manly
manor
maple
march
marry
marsh
mason
masse
match
matey
mauve
maxim
maybe
mayor
mealy
meant
meaty
mecca
medal
media
medic
melee
melon
mercy
merge
merit
merry
metal
meter
metro
micro
midge
midst
might
milky
mimic
mince
miner
minim
minor
minty
minus
mirth
miser
missy
mocha
modal
model
modem
mogul
moist
molar
moldy
money
month
moody
moose
moral
moron
morph
mossy
motel
motif
motor
motto
moult
mound
mount
mourn
mouse
mouth
mover
movie
mower
mucky
mucus
muddy
mulch
mummy
munch
mural
murky
mushy
music
musky
musty
myrrh
nadir
naive
nanny
nasal
nasty
natal
naval
navel
needy
neigh
nerdy
nerve
never
newer
newly
nicer
niche
niece
night
ninja
ninny
ninth
noble
nobly
noise
noisy
nomad
noose
north
nosey
notch
novel
nudge
nurse
nutty
nylon
nymph
oaken
obese
occur
ocean
octal
octet
odder
oddly
offal
offer
often
olden
older
olive
ombre
omega
onion
onset
opera
opine
opium
optic
orbit
order
organ
other
otter
ought
ounce
outdo
outer
outgo
ovary
ovate
overt
ovine
ovoid
owing
owner
oxide
ozone
paddy
pagan
paint
paler
palsy
panel
panic
pansy
papal
paper
parer
parka
parry
parse
party
pasta
paste
pasty
patch
patio
patsy
patty
pause
payee
payer
peace
peach
pearl
pecan
pedal
penal
pence
penne
penny
perch
peril
perky
pesky
pesto
petal
petty
phase
phone
phony
photo
piano
picky
piece
piety
piggy
pilot
pinch
piney
pinky
pinto
piper
pique
pitch
pithy
pivot
pixel
pixie
pizza
place
plaid
plain
plait
plane
plank
plant
plate
plaza
plead
pleat
plied
plier
pluck
plumb
plume
plump
plunk
plush
poesy
point
poise
poker
polar
polka
polyp
pooch
poppy
porch
poser
posit
posse
pouch
pound
pouty
power
prank
prawn
preen
press
price
prick
pride
pried
prime
primo
print
prior
prism
privy
prize
probe
prone
prong
proof
prose
proud
prove
prowl
proxy
prude
prune
psalm
pubic
pudgy
puffy
pulpy
pulse
punch
pupil
puppy
puree
purer
purge
purse
pushy
putty
pygmy
quack
quail
quake
qualm
quark
quart
quash
quasi
queen
queer
quell
query
quest
queue
quick
quiet
quill
quilt
quirk
quite
quota
quote
quoth
rabbi
rabid
racer
radar
radii
radio
rainy
raise
rajah
rally
ralph
ramen
ranch
randy
range
rapid
rarer
raspy
ratio
ratty
raven
rayon
razor
reach
react
ready
realm
rearm
rebar
rebel
rebus
rebut
recap
recur
recut
reedy
refer
refit
regal
rehab
reign
relax
relay
relic
remit
renal
renew
repay
repel
reply
rerun
reset
resin
retch
retro
retry
reuse
revel
revue
rhino
rhyme
rider
ridge
rifle
right
rigid
rigor
rinse
ripen
riper
risen
riser
risky
rival
river
rivet
roach
roast
robin
robot
rocky
rodeo
roger
rogue
roomy
roost
rotor
rouge
rough
round
rouse
route
rover
rowdy
rower
royal
ruddy
ruder
rugby
ruler
rumba
rumor
rupee
rural
rusty
sadly
safer
saint
salad
sally
salon
salsa
salty
salve
salvo
sandy
saner
sappy
sassy
satin
satyr
sauce
saucy
sauna
saute
savor
savoy
savvy
scald
scale
scalp
scaly
scamp
scant
scare
scarf
scary
scene
scent
scion
scoff
scold
scone
scoop
scope
score
scorn
scour
scout
scowl
scram
scrap
scree
screw
scrub
scrum
scuba
sedan
seedy
segue
seize
semen
sense
sepia
serif
serum
serve
setup
seven
sever
sewer
shack
shade
shady
shaft
shake
shaky
shale
shall
shalt
shame
shank
shape
shard
share
shark
sharp
shave
shawl
shear
sheen
sheep
sheer
sheet
sheik
shelf
shell
shied
shift
shine
shiny
shire
shirk
shirt
shoal
shock
shone
shook
shoot
shore
shorn
short
shout
shove
shown
showy
shrew
shrub
shrug
shuck
shunt
shush
shyly
siege
sieve
sight
sigma
silky
silly
since
sinew
singe
siren
sissy
sixth
sixty
skate
skier
skiff
skill
skimp
skirt
skulk
skull
skunk
slack
slain
slang
slant
slash
slate
sleek
sleep
sleet
slept
slice
slick
slide
slime
slimy
sling
slink
sloop
slope
slosh
sloth
slump
slung
slunk
slurp
slush
slyly
smack
small
smart
smash
smear
smell
smelt
smile
smirk
smite
smith
smock
smoke
smoky
smote
snack
snail
snake
snaky
snare
snarl
sneak
sneer
snide
sniff
snipe
snoop
snore
snort
snout
snowy
snuck
snuff
soapy
sober
soggy
solar
solid
solve
sonar
sonic
sooth
sooty
sorry
sound
south
sower
space
spade
spank
spare
spark
spasm
spawn
speak
spear
speck
speed
spell
spelt
spend
spent
sperm
spice
spicy
spied
spiel
spike
spiky
spill
spilt
spine
spiny
spire
spite
splat
split
spoil
spoke
spoof
spook
spool
spoon
spore
sport
spout
spray
spree
sprig
spunk
spurn
spurt
squad
squat
squib
stack
staff
stage
staid
stain
stair
stake
stale
stalk
stall
stamp
stand
stank
stare
stark
start
stash
state
stave
stead
steak
steal
steam
steed
steel
steep
steer
stein
stern
stick
stiff
still
stilt
sting
stink
stint
stock
stoic
stoke
stole
stomp
stone
stony
stood
stool
stoop
store
stork
storm
story
stout
stove
strap
straw
stray
strip
strut
stuck
study
stuff
stump
stung
stunk
stunt
style
suave
sugar
suing
suite
sulky
sully
sumac
sunny
super
surer
surge
surly
sushi
swami
swamp
swarm
swash
swath
swear
sweat
sweep
sweet
swell
swept
swift
swill
swine
swing
swirl
swish
swoon
swoop
sword
swore
sworn
swung
synod
syrup
tabby
table
taboo
tacit
tacky
taffy
taint
taken
taker
tally
talon
tamer
tango
tangy
taper
tapir
tardy
tarot
taste
tasty
tatty
taunt
tawny
teach
teary
tease
teddy
teeth
tempo
tenet
tenor
tense
tenth
tepee
tepid
terra
terse
testy
thank
theft
their
theme
there
these
theta
thick
thief
thigh
thing
think
third
thong
thorn
those
three
threw
throb
throw
thrum
thumb
thump
thyme
tiara
tibia
tidal
tiger
tight
tilde
timer
timid
tipsy
titan
tithe
title
toast
today
toddy
token
tonal
tonga
tonic
tooth
topaz
topic
torch
torso
torus
total
totem
touch
tough
towel
tower
toxic
toxin
trace
track
tract
trade
trail
train
trait
tramp
trash
trawl
tread
treat
trend
triad
trial
tribe
trice
trick
tried
tripe
trite
troll
troop
trope
trout
trove
truce
truck
truer
truly
trump
trunk
truss
trust
truth
tryst
tubal
tuber
tulip
tulle
tumor
tunic
turbo
tutor
twang
tweak
tweed
tweet
twice
twine
twirl
twist
twixt
tying
udder
ulcer
ultra
umbra
uncle
uncut
under
undid
undue
unfed
unfit
unify
union
unite
unity
unlit
unmet
unset
untie
until
unwed
unzip
upper
upset
urban
urine
usage
usher
using
usual
usurp
utile
utter
vague
valet
valid
valor
value
valve
vapid
vapor
vault
vaunt
vegan
venom
venue
verge
verse
verso
verve
vicar
video
vigil
vigor
villa
vinyl
viola
viper
viral
virus
visit
visor
vista
vital
vivid
vixen
vocal
vodka
vogue
voice
voila
vomit
voter
vouch
vowel
vying
wacky
wafer
wager
wagon
waist
waive
waltz
warty
waste
watch
water
waver
waxen
weary
weave
wedge
weedy
weigh
weird
welch
welsh
whack
whale
wharf
wheat
wheel
whelp
where
which
whiff
while
whine
whiny
whirl
whisk
white
whole
whoop
whose
widen
wider
widow
width
wield
wight
willy
wimpy
wince
winch
windy
wiser
wispy
witch
witty
woken
woman
women
woody
wooer
wooly
woozy
wordy
world
worry
worse
worst
worth
would
wound
woven
wrack
wrath
wreak
wreck
wrest
wring
wrist
write
wrong
wrote
wrung
wryly
yacht
yearn
yeast
yield
young
youth
zebra
zesty
zonal
"""

# ╔═╡ bb513fc5-1bdc-4c27-a778-86aa71833d0e
const wordle_original_answers = split(wordle_original_answers_raw, '\n') |> Filter(!isempty) |> Map(String) |> collect

# ╔═╡ 86616282-8287-4371-95bb-111940c16b0f
md"""
The following words are allowed guesses for the New York Times Wordle game retrieved on March 27, 2023
"""

# ╔═╡ 65995e32-cdde-49d2-a916-00a48a46ecb5
#allowed guesses embedded in the NYT wordle source code as of 05/27/2023
const nyt_valid_words = ["aahed","aalii","aapas","aargh","aarti","abaca","abaci","abacs","abaft","abaht","abaka","abamp","aband","abash","abask","abaya","abbas","abbed","abbes","abcee","abeam","abear","abeat","abeer","abele","abeng","abers","abets","abeys","abies","abius","abjad","abjud","abler","ables","ablet","ablow","abmho","abnet","abohm","aboil","aboma","aboon","abord","abore","aborn","abram","abray","abrim","abrin","abris","absey","absit","abuna","abune","abura","aburn","abuts","abuzz","abyes","abysm","acais","acara","acari","accas","accha","accoy","accra","acedy","acene","acerb","acers","aceta","achar","ached","acher","aches","achey","achoo","acids","acidy","acies","acing","acini","ackee","acker","acmes","acmic","acned","acnes","acock","acoel","acold","acone","acral","acred","acres","acron","acros","acryl","actas","acted","actin","acton","actus","acyls","adats","adawn","adaws","adays","adbot","addas","addax","added","adder","addin","addio","addle","addra","adead","adeem","adhan","adhoc","adieu","adios","adits","adlib","adman","admen","admix","adnex","adobo","adoon","adorb","adown","adoze","adrad","adraw","adred","adret","adrip","adsum","aduki","adunc","adust","advew","advts","adyta","adyts","adzed","adzes","aecia","aedes","aeger","aegis","aeons","aerie","aeros","aesir","aevum","afald","afanc","afara","afars","afear","affly","afion","afizz","aflaj","aflap","aflow","afoam","afore","afret","afrit","afros","aftos","agals","agama","agami","agamy","agars","agasp","agast","agaty","agave","agaze","agbas","agene","agers","aggag","agger","aggie","aggri","aggro","aggry","aghas","agidi","agila","agios","agism","agist","agita","aglee","aglet","agley","agloo","aglus","agmas","agoge","agogo","agone","agons","agood","agora","agria","agrin","agros","agrum","agued","agues","aguey","aguna","agush","aguti","aheap","ahent","ahigh","ahind","ahing","ahint","ahold","ahole","ahull","ahuru","aidas","aided","aides","aidoi","aidos","aiery","aigas","aight","ailed","aimag","aimak","aimed","aimer","ainee","ainga","aioli","aired","airer","airns","airth","airts","aitch","aitus","aiver","aixes","aiyah","aiyee","aiyoh","aiyoo","aizle","ajies","ajiva","ajuga","ajupa","ajwan","akara","akees","akela","akene","aking","akita","akkas","akker","akoia","akoja","akoya","aksed","akses","alaap","alack","alala","alamo","aland","alane","alang","alans","alant","alapa","alaps","alary","alata","alate","alays","albas","albee","albid","alcea","alces","alcid","alcos","aldea","alder","aldol","aleak","aleck","alecs","aleem","alefs","aleft","aleph","alews","aleye","alfas","algal","algas","algid","algin","algor","algos","algum","alias","alick","alifs","alims","aline","alios","alist","aliya","alkie","alkin","alkos","alkyd","alkyl","allan","allee","allel","allen","aller","allin","allis","allod","allus","allyl","almah","almas","almeh","almes","almud","almug","alods","aloed","aloes","aloha","aloin","aloos","alose","alowe","altho","altos","alula","alums","alumy","alure","alurk","alvar","alway","amahs","amain","amari","amaro","amate","amaut","amban","ambit","ambos","ambry","ameba","ameer","amene","amens","ament","amias","amice","amici","amide","amido","amids","amies","amiga","amigo","amine","amino","amins","amirs","amlas","amman","ammas","ammon","ammos","amnia","amnic","amnio","amoks","amole","amore","amort","amour","amove","amowt","amped","ampul","amrit","amuck","amyls","anana","anata","ancho","ancle","ancon","andic","andro","anear","anele","anent","angas","anglo","anigh","anile","anils","anima","animi","anion","anise","anker","ankhs","ankus","anlas","annal","annan","annas","annat","annum","annus","anoas","anole","anomy","ansae","ansas","antae","antar","antas","anted","antes","antis","antra","antre","antsy","anura","anyon","apace","apage","apaid","apayd","apays","apeak","apeek","apers","apert","apery","apgar","aphis","apian","apiol","apish","apism","apode","apods","apols","apoop","aport","appal","appam","appay","appel","appro","appts","appui","appuy","apres","apses","apsis","apsos","apted","apter","aquae","aquas","araba","araks","arame","arars","arbah","arbas","arced","archi","arcos","arcus","ardeb","ardri","aread","areae","areal","arear","areas","areca","aredd","arede","arefy","areic","arene","arepa","arere","arete","arets","arett","argal","argan","argil","argle","argol","argon","argot","argus","arhat","arias","ariel","ariki","arils","ariot","arish","arith","arked","arled","arles","armed","armer","armet","armil","arnas","arnis","arnut","aroba","aroha","aroid","arpas","arpen","arrah","arras","arret","arris","arroz","arsed","arses","arsey","arsis","artal","artel","arter","artic","artis","artly","aruhe","arums","arval","arvee","arvos","aryls","asada","asana","ascon","ascus","asdic","ashed","ashes","ashet","asity","askar","asked","asker","askoi","askos","aspen","asper","aspic","aspie","aspis","aspro","assai","assam","assed","asses","assez","assot","aster","astir","astun","asura","asway","aswim","asyla","ataps","ataxy","atigi","atilt","atimy","atlas","atman","atmas","atmos","atocs","atoke","atoks","atoms","atomy","atony","atopy","atria","atrip","attap","attar","attas","atter","atuas","aucht","audad","audax","augen","auger","auges","aught","aulas","aulic","auloi","aulos","aumil","aunes","aunts","aurae","aural","aurar","auras","aurei","aures","auric","auris","aurum","autos","auxin","avale","avant","avast","avels","avens","avers","avgas","avine","avion","avise","aviso","avize","avows","avyze","awari","awarn","awato","awave","aways","awdls","aweel","aweto","awing","awkin","awmry","awned","awner","awols","awork","axels","axile","axils","axing","axite","axled","axles","axman","axmen","axoid","axone","axons","ayahs","ayaya","ayelp","aygre","ayins","aymag","ayont","ayres","ayrie","azans","azide","azido","azine","azlon","azoic","azole","azons","azote","azoth","azuki","azurn","azury","azygy","azyme","azyms","baaed","baals","baaps","babas","babby","babel","babes","babka","baboo","babul","babus","bacca","bacco","baccy","bacha","bachs","backs","backy","bacne","badam","baddy","baels","baffs","baffy","bafta","bafts","baghs","bagie","bagsy","bagua","bahts","bahus","bahut","baiks","baile","bails","bairn","baisa","baith","baits","baiza","baize","bajan","bajra","bajri","bajus","baked","baken","bakes","bakra","balas","balds","baldy","baled","bales","balks","balky","ballo","balls","bally","balms","baloi","balon","baloo","balot","balsa","balti","balun","balus","balut","bamas","bambi","bamma","bammy","banak","banco","bancs","banda","bandh","bands","bandy","baned","banes","bangs","bania","banks","banky","banns","bants","bantu","banty","bantz","banya","baons","baozi","bappu","bapus","barbe","barbs","barby","barca","barde","bardo","bards","bardy","bared","barer","bares","barfi","barfs","barfy","baric","barks","barky","barms","barmy","barns","barny","barps","barra","barre","barro","barry","barye","basan","basas","based","basen","baser","bases","basha","basho","basij","basks","bason","basse","bassi","basso","bassy","basta","basti","basto","basts","bated","bates","baths","batik","batos","batta","batts","battu","bauds","bauks","baulk","baurs","bavin","bawds","bawks","bawls","bawns","bawrs","bawty","bayas","bayed","bayer","bayes","bayle","bayts","bazar","bazas","bazoo","bball","bdays","beads","beaks","beaky","beals","beams","beamy","beano","beans","beany","beare","bears","beath","beats","beaty","beaus","beaut","beaux","bebop","becap","becke","becks","bedad","bedel","bedes","bedew","bedim","bedye","beedi","beefs","beeps","beers","beery","beets","befog","begad","begar","begem","begob","begot","begum","beige","beigy","beins","beira","beisa","bekah","belah","belar","belay","belee","belga","belit","belli","bello","bells","belon","belts","belve","bemad","bemas","bemix","bemud","bends","bendy","benes","benet","benga","benis","benji","benne","benni","benny","bento","bents","benty","bepat","beray","beres","bergs","berko","berks","berme","berms","berob","beryl","besat","besaw","besee","beses","besit","besom","besot","besti","bests","betas","beted","betes","beths","betid","beton","betta","betty","bevan","bever","bevor","bevue","bevvy","bewdy","bewet","bewig","bezes","bezil","bezzy","bhais","bhaji","bhang","bhats","bhava","bhels","bhoot","bhuna","bhuts","biach","biali","bialy","bibbs","bibes","bibis","biccy","bices","bicky","bided","bider","bides","bidet","bidis","bidon","bidri","bield","biers","biffo","biffs","biffy","bifid","bigae","biggs","biggy","bigha","bight","bigly","bigos","bihon","bijou","biked","biker","bikes","bikie","bikky","bilal","bilat","bilbo","bilby","biled","biles","bilgy","bilks","bills","bimah","bimas","bimbo","binal","bindi","binds","biner","bines","bings","bingy","binit","binks","binky","bints","biogs","bions","biont","biose","biota","biped","bipod","bippy","birdo","birds","biris","birks","birle","birls","biros","birrs","birse","birsy","birze","birzz","bises","bisks","bisom","bitch","biter","bites","bitey","bitos","bitou","bitsy","bitte","bitts","bivia","bivvy","bizes","bizzo","bizzy","blabs","blads","blady","blaer","blaes","blaff","blags","blahs","blain","blams","blanc","blart","blase","blash","blate","blats","blatt","blaud","blawn","blaws","blays","bleah","blear","blebs","blech","blees","blent","blert","blest","blets","bleys","blimy","bling","blini","blins","bliny","blips","blist","blite","blits","blive","blobs","blocs","blogs","blonx","blook","bloop","blore","blots","blows","blowy","blubs","blude","bluds","bludy","blued","blues","bluet","bluey","bluid","blume","blunk","blurs","blype","boabs","boaks","boars","boart","boats","boaty","bobac","bobak","bobas","bobol","bobos","bocca","bocce","bocci","boche","bocks","boded","bodes","bodge","bodgy","bodhi","bodle","bodoh","boeps","boers","boeti","boets","boeuf","boffo","boffs","bogan","bogey","boggy","bogie","bogle","bogue","bogus","bohea","bohos","boils","boing","boink","boite","boked","bokeh","bokes","bokos","bolar","bolas","boldo","bolds","boles","bolet","bolix","bolks","bolls","bolos","bolts","bolus","bomas","bombe","bombo","bombs","bomoh","bomor","bonce","bonds","boned","boner","bones","bongs","bonie","bonks","bonne","bonny","bonum","bonza","bonze","booai","booay","boobs","boody","booed","boofy","boogy","boohs","books","booky","bools","booms","boomy","boong","boons","boord","boors","boose","boots","boppy","borak","boral","boras","borde","bords","bored","boree","borek","borel","borer","bores","borgo","boric","borks","borms","borna","boron","borts","borty","bortz","bosey","bosie","bosks","bosky","boson","bossa","bosun","botas","boteh","botel","botes","botew","bothy","botos","botte","botts","botty","bouge","bouks","boult","bouns","bourd","bourg","bourn","bouse","bousy","bouts","boutu","bovid","bowat","bowed","bower","bowes","bowet","bowie","bowls","bowne","bowrs","bowse","boxed","boxen","boxes","boxla","boxty","boyar","boyau","boyed","boyey","boyfs","boygs","boyla","boyly","boyos","boysy","bozos","braai","brach","brack","bract","brads","braes","brags","brahs","brail","braks","braky","brame","brane","brank","brans","brant","brast","brats","brava","bravi","braws","braxy","brays","braza","braze","bream","brede","breds","breem","breer","brees","breid","breis","breme","brens","brent","brere","brers","breve","brews","breys","brier","bries","brigs","briki","briks","brill","brims","brins","brios","brise","briss","brith","brits","britt","brize","broch","brock","brods","brogh","brogs","brome","bromo","bronc","brond","brool","broos","brose","brosy","brows","bruck","brugh","bruhs","bruin","bruit","bruja","brujo","brule","brume","brung","brusk","brust","bruts","bruvs","buats","buaze","bubal","bubas","bubba","bubbe","bubby","bubus","buchu","bucko","bucks","bucku","budas","buded","budes","budis","budos","buena","buffa","buffe","buffi","buffo","buffs","buffy","bufos","bufty","bugan","buhls","buhrs","buiks","buist","bukes","bukos","bulbs","bulgy","bulks","bulla","bulls","bulse","bumbo","bumfs","bumph","bumps","bumpy","bunas","bunce","bunco","bunde","bundh","bunds","bundt","bundu","bundy","bungs","bungy","bunia","bunje","bunjy","bunko","bunks","bunns","bunts","bunty","bunya","buoys","buppy","buran","buras","burbs","burds","buret","burfi","burgh","burgs","burin","burka","burke","burks","burls","burns","buroo","burps","burqa","burra","burro","burrs","burry","bursa","burse","busby","buses","busks","busky","bussu","busti","busts","busty","buteo","butes","butle","butoh","butts","butty","butut","butyl","buyin","buzzy","bwana","bwazi","byded","bydes","byked","bykes","byres","byrls","byssi","bytes","byway","caaed","cabas","caber","cabob","caboc","cabre","cacas","cacks","cacky","cadee","cades","cadge","cadgy","cadie","cadis","cadre","caeca","caese","cafes","caffe","caffs","caged","cager","cages","cagot","cahow","caids","cains","caird","cajon","cajun","caked","cakes","cakey","calfs","calid","calif","calix","calks","calla","calle","calls","calms","calmy","calos","calpa","calps","calve","calyx","caman","camas","cames","camis","camos","campi","campo","camps","campy","camus","cando","caned","caneh","caner","canes","cangs","canid","canna","canns","canso","canst","canti","canto","cants","canty","capas","capax","caped","capes","capex","caphs","capiz","caple","capon","capos","capot","capri","capul","carap","carbo","carbs","carby","cardi","cards","cardy","cared","carer","cares","caret","carex","carks","carle","carls","carne","carns","carny","carob","carom","caron","carpe","carpi","carps","carrs","carse","carta","carte","carts","carvy","casas","casco","cased","caser","cases","casks","casky","casts","casus","cates","cauda","cauks","cauld","cauls","caums","caups","cauri","causa","cavas","caved","cavel","caver","caves","cavie","cavus","cawed","cawks","caxon","ceaze","cebid","cecal","cecum","ceded","ceder","cedes","cedis","ceiba","ceili","ceils","celeb","cella","celli","cells","celly","celom","celts","cense","cento","cents","centu","ceorl","cepes","cerci","cered","ceres","cerge","ceria","ceric","cerne","ceroc","ceros","certs","certy","cesse","cesta","cesti","cetes","cetyl","cezve","chaap","chaat","chace","chack","chaco","chado","chads","chaft","chais","chals","chams","chana","chang","chank","chape","chaps","chapt","chara","chare","chark","charr","chars","chary","chats","chava","chave","chavs","chawk","chawl","chaws","chaya","chays","cheba","chedi","cheeb","cheep","cheet","chefs","cheka","chela","chelp","chemo","chems","chere","chert","cheth","chevy","chews","chewy","chiao","chias","chiba","chibs","chica","chich","chico","chics","chiel","chiko","chiks","chile","chimb","chimo","chimp","chine","ching","chink","chino","chins","chips","chirk","chirl","chirm","chiro","chirr","chirt","chiru","chiti","chits","chiva","chive","chivs","chivy","chizz","choco","chocs","chode","chogs","choil","choko","choky","chola","choli","cholo","chomp","chons","choof","chook","choom","choon","chops","choss","chota","chott","chout","choux","chowk","chows","chubs","chufa","chuff","chugs","chums","churl","churr","chuse","chuts","chyle","chyme","chynd","cibol","cided","cides","ciels","ciggy","cilia","cills","cimar","cimex","cinct","cines","cinqs","cions","cippi","circs","cires","cirls","cirri","cisco","cissy","cists","cital","cited","citee","citer","cites","cives","civet","civie","civvy","clach","clade","clads","claes","clags","clair","clame","clams","clans","claps","clapt","claro","clart","clary","clast","clats","claut","clave","clavi","claws","clays","cleck","cleek","cleep","clefs","clegs","cleik","clems","clepe","clept","cleve","clews","clied","clies","clift","clime","cline","clint","clipe","clips","clipt","clits","cloam","clods","cloff","clogs","cloke","clomb","clomp","clonk","clons","cloop","cloot","clops","clote","clots","clour","clous","clows","cloye","cloys","cloze","clubs","clues","cluey","clunk","clype","cnida","coact","coady","coala","coals","coaly","coapt","coarb","coate","coati","coats","cobbs","cobby","cobia","coble","cobot","cobza","cocas","cocci","cocco","cocks","cocky","cocos","cocus","codas","codec","coded","coden","coder","codes","codex","codon","coeds","coffs","cogie","cogon","cogue","cohab","cohen","cohoe","cohog","cohos","coifs","coign","coils","coins","coirs","coits","coked","cokes","cokey","colas","colby","colds","coled","coles","coley","colic","colin","colle","colls","colly","colog","colts","colza","comae","comal","comas","combe","combi","combo","combs","comby","comer","comes","comix","comme","commo","comms","commy","compo","comps","compt","comte","comus","coned","cones","conex","coney","confs","conga","conge","congo","conia","conin","conks","conky","conne","conns","conte","conto","conus","convo","cooch","cooed","cooee","cooer","cooey","coofs","cooks","cooky","cools","cooly","coomb","cooms","coomy","coons","coops","coopt","coost","coots","cooty","cooze","copal","copay","coped","copen","coper","copes","copha","coppy","copra","copsy","coqui","coram","corbe","corby","corda","cords","cored","cores","corey","corgi","coria","corks","corky","corms","corni","corno","corns","cornu","corps","corse","corso","cosec","cosed","coses","coset","cosey","cosie","costa","coste","costs","cotan","cotch","coted","cotes","coths","cotta","cotts","coude","coups","courb","courd","coure","cours","couta","couth","coved","coves","covin","cowal","cowan","cowed","cowks","cowls","cowps","cowry","coxae","coxal","coxed","coxes","coxib","coyau","coyed","coyer","coypu","cozed","cozen","cozes","cozey","cozie","craal","crabs","crags","craic","craig","crake","crame","crams","crans","crape","craps","crapy","crare","craws","crays","creds","creel","crees","crein","crema","crems","crena","creps","crepy","crewe","crews","crias","cribo","cribs","cries","crims","crine","crink","crins","crios","cripe","crips","crise","criss","crith","crits","croci","crocs","croft","crogs","cromb","crome","cronk","crons","crool","croon","crops","crore","crost","crout","crowl","crows","croze","cruck","crudo","cruds","crudy","crues","cruet","cruft","crunk","cruor","crura","cruse","crusy","cruve","crwth","cryer","cryne","ctene","cubby","cubeb","cubed","cuber","cubes","cubit","cucks","cudda","cuddy","cueca","cuffo","cuffs","cuifs","cuing","cuish","cuits","cukes","culch","culet","culex","culls","cully","culms","culpa","culti","cults","culty","cumec","cundy","cunei","cunit","cunny","cunts","cupel","cupid","cuppa","cuppy","cupro","curat","curbs","curch","curds","curdy","cured","curer","cures","curet","curfs","curia","curie","curli","curls","curns","curny","currs","cursi","curst","cusec","cushy","cusks","cusps","cuspy","cusso","cusum","cutch","cuter","cutes","cutey","cutin","cutis","cutto","cutty","cutup","cuvee","cuzes","cwtch","cyano","cyans","cycad","cycas","cyclo","cyder","cylix","cymae","cymar","cymas","cymes","cymol","cysts","cytes","cyton","czars","daals","dabba","daces","dacha","dacks","dadah","dadas","dadis","dadla","dados","daffs","daffy","dagga","daggy","dagos","dahis","dahls","daiko","daine","daint","daker","daled","dalek","dales","dalis","dalle","dalts","daman","damar","dames","damme","damna","damns","damps","dampy","dancy","danda","dangs","danio","danks","danny","danse","dants","dappy","daraf","darbs","darcy","dared","darer","dares","darga","dargs","daric","daris","darks","darky","darls","darns","darre","darts","darzi","dashi","dashy","datal","dated","dater","dates","datil","datos","datto","daube","daubs","dauby","dauds","dault","daurs","dauts","daven","davit","dawah","dawds","dawed","dawen","dawgs","dawks","dawns","dawts","dayal","dayan","daych","daynt","dazed","dazer","dazes","dbags","deads","deair","deals","deans","deare","dearn","dears","deary","deash","deave","deaws","deawy","debag","debby","debel","debes","debts","debud","debur","debus","debye","decad","decaf","decan","decim","decko","decks","decos","decyl","dedal","deeds","deedy","deely","deems","deens","deeps","deere","deers","deets","deeve","deevs","defat","deffo","defis","defog","degas","degum","degus","deice","deids","deify","deils","deink","deism","deist","deked","dekes","dekko","deled","deles","delfs","delft","delis","della","dells","delly","delos","delph","delts","deman","demes","demic","demit","demob","demoi","demos","demot","dempt","denar","denay","dench","denes","denet","denis","dente","dents","deoch","deoxy","derat","deray","dered","deres","derig","derma","derms","derns","derny","deros","derpy","derro","derry","derth","dervs","desex","deshi","desis","desks","desse","detag","devas","devel","devis","devon","devos","devot","dewan","dewar","dewax","dewed","dexes","dexie","dexys","dhaba","dhaks","dhals","dhikr","dhobi","dhole","dholl","dhols","dhoni","dhoti","dhows","dhuti","diact","dials","diana","diane","diazo","dibbs","diced","dicer","dices","dicht","dicks","dicky","dicot","dicta","dicto","dicts","dictu","dicty","diddy","didie","didis","didos","didst","diebs","diels","diene","diets","diffs","dight","dikas","diked","diker","dikes","dikey","dildo","dilli","dills","dimbo","dimer","dimes","dimps","dinar","dined","dines","dinge","dings","dinic","dinks","dinky","dinlo","dinna","dinos","dints","dioch","diols","diota","dippy","dipso","diram","direr","dirke","dirks","dirls","dirts","disas","disci","discs","dishy","disks","disme","dital","ditas","dited","dites","ditsy","ditts","ditzy","divan","divas","dived","dives","divey","divis","divna","divos","divot","divvy","diwan","dixie","dixit","diyas","dizen","djinn","djins","doabs","doats","dobby","dobes","dobie","dobla","doble","dobra","dobro","docht","docks","docos","docus","doddy","dodos","doeks","doers","doest","doeth","doffs","dogal","dogan","doges","dogey","doggo","doggy","dogie","dogly","dohyo","doilt","doily","doits","dojos","dolce","dolci","doled","dolee","doles","doley","dolia","dolie","dolls","dolma","dolor","dolos","dolts","domal","domed","domes","domic","donah","donas","donee","doner","donga","dongs","donko","donna","donne","donny","donsy","doobs","dooce","doody","doofs","dooks","dooky","doole","dools","dooly","dooms","doomy","doona","doorn","doors","doozy","dopas","doped","doper","dopes","doppe","dorad","dorba","dorbs","doree","dores","doric","doris","dorje","dorks","dorky","dorms","dormy","dorps","dorrs","dorsa","dorse","dorts","dorty","dosai","dosas","dosed","doseh","doser","doses","dosha","dotal","doted","doter","dotes","dotty","douar","douce","doucs","douks","doula","douma","doums","doups","doura","douse","douts","doved","doven","dover","doves","dovie","dowak","dowar","dowds","dowed","dower","dowfs","dowie","dowle","dowls","dowly","downa","downs","dowps","dowse","dowts","doxed","doxes","doxie","doyen","doyly","dozed","dozer","dozes","drabs","drack","draco","draff","drags","drail","drams","drant","draps","drapy","drats","drave","draws","drays","drear","dreck","dreed","dreer","drees","dregs","dreks","drent","drere","drest","dreys","dribs","drice","dries","drily","drips","dript","drock","droid","droil","droke","drole","drome","drony","droob","droog","drook","drops","dropt","drouk","drows","drubs","drugs","drums","drupe","druse","drusy","druxy","dryad","dryas","dsobo","dsomo","duads","duals","duans","duars","dubbo","dubby","ducal","ducat","duces","ducks","ducky","ducti","ducts","duddy","duded","dudes","duels","duets","duett","duffs","dufus","duing","duits","dukas","duked","dukes","dukka","dukun","dulce","dules","dulia","dulls","dulse","dumas","dumbo","dumbs","dumka","dumky","dumps","dunam","dunch","dunes","dungs","dungy","dunks","dunno","dunny","dunsh","dunts","duomi","duomo","duped","duper","dupes","duple","duply","duppy","dural","duras","dured","dures","durgy","durns","duroc","duros","duroy","durra","durrs","durry","durst","durum","durzi","dusks","dusts","duxes","dwaal","dwale","dwalm","dwams","dwamy","dwang","dwaum","dweeb","dwile","dwine","dyads","dyers","dyked","dykes","dykey","dykon","dynel","dynes","dynos","dzhos","eagly","eagre","ealed","eales","eaned","eards","eared","earls","earns","earnt","earst","eased","easer","eases","easle","easts","eathe","eatin","eaved","eaver","eaves","ebank","ebbed","ebbet","ebena","ebene","ebike","ebons","ebook","ecads","ecard","ecash","eched","eches","echos","ecigs","ecole","ecrus","edema","edged","edger","edges","edile","edits","educe","educt","eejit","eensy","eeven","eever","eevns","effed","effer","efits","egads","egers","egest","eggar","egged","egger","egmas","ehing","eider","eidos","eigne","eiked","eikon","eilds","eiron","eisel","ejido","ekdam","ekkas","elain","eland","elans","elchi","eldin","eleet","elemi","elfed","eliad","elint","elmen","eloge","elogy","eloin","elops","elpee","elsin","elute","elvan","elven","elver","elves","emacs","embar","embay","embog","embow","embox","embus","emeer","emend","emerg","emery","emeus","emics","emirs","emits","emmas","emmer","emmet","emmew","emmys","emoji","emong","emote","emove","empts","emule","emure","emyde","emyds","enarm","enate","ended","ender","endew","endue","enews","enfix","eniac","enlit","enmew","ennog","enoki","enols","enorm","enows","enrol","ensew","ensky","entia","entre","enure","enurn","envoi","enzym","eolid","eorls","eosin","epact","epees","epena","epene","ephah","ephas","ephod","ephor","epics","epode","epopt","eppie","epris","eques","equid","erbia","erevs","ergon","ergos","ergot","erhus","erica","erick","erics","ering","erned","ernes","erose","erred","erses","eruct","erugo","eruvs","erven","ervil","escar","escot","esile","eskar","esker","esnes","esrog","esses","estoc","estop","estro","etage","etape","etats","etens","ethal","ethne","ethyl","etics","etnas","etrog","ettin","ettle","etuis","etwee","etyma","eughs","euked","eupad","euros","eusol","evegs","evens","evert","evets","evhoe","evils","evite","evohe","ewers","ewest","ewhow","ewked","exams","exeat","execs","exeem","exeme","exfil","exier","exies","exine","exing","exite","exits","exode","exome","exons","expat","expos","exude","exuls","exurb","eyass","eyers","eyots","eyras","eyres","eyrie","eyrir","ezine","fabbo","fabby","faced","facer","faces","facey","facia","facie","facta","facto","facts","facty","faddy","faded","fader","fades","fadge","fados","faena","faery","faffs","faffy","faggy","fagin","fagot","faiks","fails","faine","fains","faire","fairs","faked","faker","fakes","fakey","fakie","fakir","falaj","fales","falls","falsy","famed","fames","fanal","fands","fanes","fanga","fango","fangs","fanks","fanon","fanos","fanum","faqir","farad","farci","farcy","fards","fared","farer","fares","farle","farls","farms","faros","farro","farse","farts","fasci","fasti","fasts","fated","fates","fatly","fatso","fatwa","fauch","faugh","fauld","fauns","faurd","faute","fauts","fauve","favas","favel","faver","faves","favus","fawns","fawny","faxed","faxes","fayed","fayer","fayne","fayre","fazed","fazes","feals","feard","feare","fears","feart","fease","feats","feaze","feces","fecht","fecit","fecks","fedai","fedex","feebs","feeds","feels","feely","feens","feers","feese","feeze","fehme","feint","feist","felch","felid","felix","fells","felly","felts","felty","femal","femes","femic","femmy","fends","fendy","fenis","fenks","fenny","fents","feods","feoff","ferer","feres","feria","ferly","fermi","ferms","ferns","ferny","ferox","fesse","festa","fests","festy","fetas","feted","fetes","fetor","fetta","fetts","fetwa","feuar","feuds","feued","feyed","feyer","feyly","fezes","fezzy","fiars","fiats","fibre","fibro","fices","fiche","fichu","ficin","ficos","ficta","fides","fidge","fidos","fidus","fiefs","fient","fiere","fieri","fiers","fiest","fifed","fifer","fifes","fifis","figgy","figos","fiked","fikes","filar","filch","filed","files","filii","filks","fille","fillo","fills","filmi","films","filon","filos","filum","finca","finds","fined","fines","finis","finks","finny","finos","fiord","fiqhs","fique","fired","firer","fires","firie","firks","firma","firms","firni","firns","firry","firth","fiscs","fisho","fisks","fists","fisty","fitch","fitly","fitna","fitte","fitts","fiver","fives","fixed","fixes","fixie","fixit","fjeld","flabs","flaff","flags","flaks","flamm","flams","flamy","flane","flans","flaps","flary","flats","flava","flawn","flaws","flawy","flaxy","flays","fleam","fleas","fleek","fleer","flees","flegs","fleme","fleur","flews","flexi","flexo","fleys","flics","flied","flies","flimp","flims","flips","flirs","flisk","flite","flits","flitt","flobs","flocs","floes","flogs","flong","flops","flore","flors","flory","flosh","flota","flote","flows","flowy","flubs","flued","flues","fluey","fluky","flump","fluor","flurr","fluty","fluyt","flyby","flyin","flype","flyte","fnarr","foals","foams","foehn","fogey","fogie","fogle","fogos","fogou","fohns","foids","foils","foins","folds","foley","folia","folic","folie","folks","folky","fomes","fonda","fonds","fondu","fones","fonio","fonly","fonts","foods","foody","fools","foots","footy","foram","forbs","forby","fordo","fords","forel","fores","forex","forks","forky","forma","forme","forms","forts","forza","forze","fossa","fosse","fouat","fouds","fouer","fouet","foule","fouls","fount","fours","fouth","fovea","fowls","fowth","foxed","foxes","foxie","foyle","foyne","frabs","frack","fract","frags","fraim","frais","franc","frape","fraps","frass","frate","frati","frats","fraus","frays","frees","freet","freit","fremd","frena","freon","frere","frets","fribs","frier","fries","frigs","frise","frist","frita","frite","frith","frits","fritt","frize","frizz","froes","frogs","fromm","frons","froom","frore","frorn","frory","frosh","frows","frowy","froyo","frugs","frump","frush","frust","fryer","fubar","fubby","fubsy","fucks","fucus","fuddy","fudgy","fuels","fuero","fuffs","fuffy","fugal","fuggy","fugie","fugio","fugis","fugle","fugly","fugus","fujis","fulla","fulls","fulth","fulwa","fumed","fumer","fumes","fumet","funda","fundi","fundo","funds","fundy","fungo","fungs","funic","funis","funks","funsy","funts","fural","furan","furca","furls","furol","furos","furrs","furth","furze","furzy","fused","fusee","fusel","fuses","fusil","fusks","fusts","fusty","futon","fuzed","fuzee","fuzes","fuzil","fyces","fyked","fykes","fyles","fyrds","fytte","gabba","gabby","gable","gaddi","gades","gadge","gadgy","gadid","gadis","gadje","gadjo","gadso","gaffs","gaged","gager","gages","gaids","gains","gairs","gaita","gaits","gaitt","gajos","galah","galas","galax","galea","galed","gales","galia","galis","galls","gally","galop","galut","galvo","gamas","gamay","gamba","gambe","gambo","gambs","gamed","games","gamey","gamic","gamin","gamme","gammy","gamps","ganch","gandy","ganef","ganev","gangs","ganja","ganks","ganof","gants","gaols","gaped","gaper","gapes","gapos","gappy","garam","garba","garbe","garbo","garbs","garda","garde","gares","garis","garms","garni","garre","garri","garth","garum","gases","gashy","gasps","gaspy","gasts","gatch","gated","gater","gates","gaths","gator","gauch","gaucy","gauds","gauje","gault","gaums","gaumy","gaups","gaurs","gauss","gauzy","gavot","gawcy","gawds","gawks","gawps","gawsy","gayal","gazal","gazar","gazed","gazes","gazon","gazoo","geals","geans","geare","gears","geasa","geats","gebur","gecks","geeks","geeps","geest","geist","geits","gelds","gelee","gelid","gelly","gelts","gemel","gemma","gemmy","gemot","genae","genal","genas","genes","genet","genic","genii","genin","genio","genip","genny","genoa","genom","genro","gents","genty","genua","genus","geode","geoid","gerah","gerbe","geres","gerle","germs","germy","gerne","gesse","gesso","geste","gests","getas","getup","geums","geyan","geyer","ghast","ghats","ghaut","ghazi","ghees","ghest","ghusl","ghyll","gibed","gibel","giber","gibes","gibli","gibus","gifts","gigas","gighe","gigot","gigue","gilas","gilds","gilet","gilia","gills","gilly","gilpy","gilts","gimel","gimme","gimps","gimpy","ginch","ginga","ginge","gings","ginks","ginny","ginzo","gipon","gippo","gippy","girds","girlf","girls","girns","giron","giros","girrs","girsh","girts","gismo","gisms","gists","gitch","gites","giust","gived","gives","gizmo","glace","glads","glady","glaik","glair","glamp","glams","glans","glary","glatt","glaum","glaur","glazy","gleba","glebe","gleby","glede","gleds","gleed","gleek","glees","gleet","gleis","glens","glent","gleys","glial","glias","glibs","gliff","glift","glike","glime","glims","glisk","glits","glitz","gloam","globi","globs","globy","glode","glogg","gloms","gloop","glops","glost","glout","glows","glowy","gloze","glued","gluer","glues","gluey","glugg","glugs","glume","glums","gluon","glute","gluts","gnapi","gnarl","gnarr","gnars","gnats","gnawn","gnaws","gnows","goads","goafs","goaft","goals","goary","goats","goaty","goave","goban","gobar","gobbe","gobbi","gobbo","gobby","gobis","gobos","godet","godso","goels","goers","goest","goeth","goety","gofer","goffs","gogga","gogos","goier","gojis","gokes","golds","goldy","goles","golfs","golpe","golps","gombo","gomer","gompa","gonch","gonef","gongs","gonia","gonif","gonks","gonna","gonof","gonys","gonzo","gooby","goodo","goods","goofs","googs","gooks","gooky","goold","gools","gooly","goomy","goons","goony","goops","goopy","goors","goory","goosy","gopak","gopik","goral","goras","goray","gorbs","gordo","gored","gores","goris","gorms","gormy","gorps","gorse","gorsy","gosht","gosse","gotch","goths","gothy","gotta","gouch","gouks","goura","gouts","gouty","goved","goves","gowan","gowds","gowfs","gowks","gowls","gowns","goxes","goyim","goyle","graal","grabs","grads","graff","graip","grama","grame","gramp","grams","grana","grano","grans","grapy","grata","grats","gravs","grays","grebe","grebo","grece","greek","grees","grege","grego","grein","grens","greps","grese","greve","grews","greys","grice","gride","grids","griff","grift","grigs","grike","grins","griot","grips","gript","gripy","grise","grist","grisy","grith","grits","grize","groat","grody","grogs","groks","groma","groms","grone","groof","grosz","grots","grouf","grovy","grows","grrls","grrrl","grubs","grued","grues","grufe","grume","grump","grund","gryce","gryde","gryke","grype","grypt","guaco","guana","guano","guans","guars","gubba","gucks","gucky","gudes","guffs","gugas","guggl","guido","guids","guimp","guiro","gulab","gulag","gular","gulas","gules","gulet","gulfs","gulfy","gulls","gulph","gulps","gulpy","gumma","gummi","gumps","gunas","gundi","gundy","gunge","gungy","gunks","gunky","gunny","guqin","gurdy","gurge","gurks","gurls","gurly","gurns","gurry","gursh","gurus","gushy","gusla","gusle","gusli","gussy","gusts","gutsy","gutta","gutty","guyed","guyle","guyot","guyse","gwine","gyals","gyans","gybed","gybes","gyeld","gymps","gynae","gynie","gynny","gynos","gyoza","gypes","gypos","gyppo","gyppy","gyral","gyred","gyres","gyron","gyros","gyrus","gytes","gyved","gyver","gyves","haafs","haars","haats","hable","habus","hacek","hacks","hacky","hadal","haded","hades","hadji","hadst","haems","haere","haets","haffs","hafiz","hafta","hafts","haggs","haham","hahas","haick","haika","haiks","haiku","hails","haily","hains","haint","hairs","haith","hajes","hajis","hajji","hakam","hakas","hakea","hakes","hakim","hakus","halal","haldi","haled","haler","hales","halfa","halfs","halid","hallo","halls","halma","halms","halon","halos","halse","halsh","halts","halva","halwa","hamal","hamba","hamed","hamel","hames","hammy","hamza","hanap","hance","hanch","handi","hands","hangi","hangs","hanks","hanky","hansa","hanse","hants","haole","haoma","hapas","hapax","haply","happi","hapus","haram","hards","hared","hares","harim","harks","harls","harms","harns","haros","harps","harts","hashy","hasks","hasps","hasta","hated","hates","hatha","hathi","hatty","hauds","haufs","haugh","haugo","hauld","haulm","hauls","hault","hauns","hause","havan","havel","haver","haves","hawed","hawks","hawms","hawse","hayed","hayer","hayey","hayle","hazan","hazed","hazer","hazes","hazle","heads","heald","heals","heame","heaps","heapy","heare","hears","heast","heats","heaty","heben","hebes","hecht","hecks","heder","hedgy","heeds","heedy","heels","heeze","hefte","hefts","heiau","heids","heigh","heils","heirs","hejab","hejra","heled","heles","helio","hella","hells","helly","helms","helos","helot","helps","helve","hemal","hemes","hemic","hemin","hemps","hempy","hench","hends","henge","henna","henny","henry","hents","hepar","herbs","herby","herds","heres","herls","herma","herms","herns","heros","herps","herry","herse","hertz","herye","hesps","hests","hetes","heths","heuch","heugh","hevea","hevel","hewed","hewer","hewgh","hexad","hexed","hexer","hexes","hexyl","heyed","hiant","hibas","hicks","hided","hider","hides","hiems","hifis","highs","hight","hijab","hijra","hiked","hiker","hikes","hikoi","hilar","hilch","hillo","hills","hilsa","hilts","hilum","hilus","himbo","hinau","hinds","hings","hinky","hinny","hints","hiois","hiped","hiper","hipes","hiply","hired","hiree","hirer","hires","hissy","hists","hithe","hived","hiver","hives","hizen","hoach","hoaed","hoagy","hoars","hoary","hoast","hobos","hocks","hocus","hodad","hodja","hoers","hogan","hogen","hoggs","hoghs","hogoh","hogos","hohed","hoick","hoied","hoiks","hoing","hoise","hokas","hoked","hokes","hokey","hokis","hokku","hokum","holds","holed","holes","holey","holks","holla","hollo","holme","holms","holon","holos","holts","homas","homed","homes","homey","homie","homme","homos","honan","honda","honds","honed","honer","hones","hongi","hongs","honks","honky","hooch","hoods","hoody","hooey","hoofs","hoogo","hooha","hooka","hooks","hooky","hooly","hoons","hoops","hoord","hoors","hoosh","hoots","hooty","hoove","hopak","hoped","hoper","hopes","hoppy","horah","horal","horas","horis","horks","horme","horns","horst","horsy","hosed","hosel","hosen","hoser","hoses","hosey","hosta","hosts","hotch","hoten","hotis","hotte","hotty","houff","houfs","hough","houri","hours","houts","hovea","hoved","hoven","hoves","howay","howbe","howes","howff","howfs","howks","howls","howre","howso","howto","hoxed","hoxes","hoyas","hoyed","hoyle","hubba","hubby","hucks","hudna","hudud","huers","huffs","huffy","huger","huggy","huhus","huias","huies","hukou","hulas","hules","hulks","hulky","hullo","hulls","hully","humas","humfs","humic","humps","humpy","hundo","hunks","hunts","hurds","hurls","hurly","hurra","hurst","hurts","hurty","hushy","husks","husos","hutia","huzza","huzzy","hwyls","hydel","hydra","hyens","hygge","hying","hykes","hylas","hyleg","hyles","hylic","hymns","hynde","hyoid","hyped","hypes","hypha","hyphy","hypos","hyrax","hyson","hythe","iambi","iambs","ibrik","icers","iched","iches","ichor","icier","icker","ickle","icons","ictal","ictic","ictus","idant","iddah","iddat","iddut","ideas","idees","ident","idled","idles","idlis","idola","idols","idyls","iftar","igapo","igged","iglus","ignis","ihram","iiwis","ikans","ikats","ikons","ileac","ileal","ileum","ileus","iliad","ilial","ilium","iller","illth","imago","imagy","imams","imari","imaum","imbar","imbed","imbos","imide","imido","imids","imine","imino","imlis","immew","immit","immix","imped","impis","impot","impro","imshi","imshy","inapt","inarm","inbye","incas","incel","incle","incog","incus","incut","indew","india","indie","indol","indow","indri","indue","inerm","infix","infos","infra","ingan","ingle","inion","inked","inker","inkle","inned","innie","innit","inorb","inros","inrun","insee","inset","inspo","intel","intil","intis","intra","inula","inure","inurn","inust","invar","inver","inwit","iodic","iodid","iodin","ioras","iotas","ippon","irade","irids","iring","irked","iroko","irone","irons","isbas","ishes","isled","isles","isnae","issei","istle","items","ither","ivied","ivies","ixias","ixnay","ixora","ixtle","izard","izars","izzat","jaaps","jabot","jacal","jacet","jacks","jacky","jaded","jades","jafas","jaffa","jagas","jager","jaggs","jaggy","jagir","jagra","jails","jaker","jakes","jakey","jakie","jalap","jaleo","jalop","jambe","jambo","jambs","jambu","james","jammy","jamon","jamun","janes","janky","janns","janny","janty","japan","japed","japer","japes","jarks","jarls","jarps","jarta","jarul","jasey","jaspe","jasps","jatha","jatis","jatos","jauks","jaune","jaups","javas","javel","jawan","jawed","jawns","jaxie","jeans","jeats","jebel","jedis","jeels","jeely","jeeps","jeera","jeers","jeeze","jefes","jeffs","jehad","jehus","jelab","jello","jells","jembe","jemmy","jenny","jeons","jerid","jerks","jerry","jesse","jessy","jests","jesus","jetee","jetes","jeton","jeune","jewed","jewie","jhala","jheel","jhils","jiaos","jibba","jibbs","jibed","jiber","jibes","jiffs","jiggy","jigot","jihad","jills","jilts","jimmy","jimpy","jingo","jings","jinks","jinne","jinni","jinns","jirds","jirga","jirre","jisms","jitis","jitty","jived","jiver","jives","jivey","jnana","jobed","jobes","jocko","jocks","jocky","jocos","jodel","joeys","johns","joins","joked","jokes","jokey","jokol","joled","joles","jolie","jollo","jolls","jolts","jolty","jomon","jomos","jones","jongs","jonty","jooks","joram","jorts","jorum","jotas","jotty","jotun","joual","jougs","jouks","joule","jours","jowar","jowed","jowls","jowly","joyed","jubas","jubes","jucos","judas","judgy","judos","jugal","jugum","jujus","juked","jukes","jukus","julep","julia","jumar","jumby","jumps","junco","junks","junky","jupes","jupon","jural","jurat","jurel","jures","juris","juste","justs","jutes","jutty","juves","juvie","kaama","kabab","kabar","kabob","kacha","kacks","kadai","kades","kadis","kafir","kagos","kagus","kahal","kaiak","kaids","kaies","kaifs","kaika","kaiks","kails","kaims","kaing","kains","kajal","kakas","kakis","kalam","kalas","kales","kalif","kalis","kalpa","kalua","kamas","kames","kamik","kamis","kamme","kanae","kanal","kanas","kanat","kandy","kaneh","kanes","kanga","kangs","kanji","kants","kanzu","kaons","kapai","kapas","kapha","kaphs","kapok","kapow","kapur","kapus","kaput","karai","karas","karat","karee","karez","karks","karns","karoo","karos","karri","karst","karsy","karts","karzy","kasha","kasme","katal","katas","katis","katti","kaugh","kauri","kauru","kaury","kaval","kavas","kawas","kawau","kawed","kayle","kayos","kazis","kazoo","kbars","kcals","keaki","kebar","kebob","kecks","kedge","kedgy","keech","keefs","keeks","keels","keema","keeno","keens","keeps","keets","keeve","kefir","kehua","keirs","kelep","kelim","kells","kelly","kelps","kelpy","kelts","kelty","kembo","kembs","kemps","kempt","kempy","kenaf","kench","kendo","kenos","kente","kents","kepis","kerbs","kerel","kerfs","kerky","kerma","kerne","kerns","keros","kerry","kerve","kesar","kests","ketas","ketch","ketes","ketol","kevel","kevil","kexes","keyed","keyer","khadi","khads","khafs","khana","khans","khaph","khats","khaya","khazi","kheda","kheer","kheth","khets","khirs","khoja","khors","khoum","khuds","khula","khyal","kiaat","kiack","kiaki","kiang","kiasu","kibbe","kibbi","kibei","kibes","kibla","kicks","kicky","kiddo","kiddy","kidel","kideo","kidge","kiefs","kiers","kieve","kievs","kight","kikay","kikes","kikoi","kiley","kilig","kilim","kills","kilns","kilos","kilps","kilts","kilty","kimbo","kimet","kinas","kinda","kinds","kindy","kines","kings","kingy","kinin","kinks","kinos","kiore","kipah","kipas","kipes","kippa","kipps","kipsy","kirby","kirks","kirns","kirri","kisan","kissy","kists","kitab","kited","kiter","kites","kithe","kiths","kitke","kitul","kivas","kiwis","klang","klaps","klett","klick","klieg","kliks","klong","kloof","kluge","klutz","knags","knaps","knarl","knars","knaur","knawe","knees","knell","knick","knish","knits","knive","knobs","knoop","knops","knosp","knots","knoud","knout","knowd","knowe","knows","knubs","knule","knurl","knurr","knurs","knuts","koans","koaps","koban","kobos","koels","koffs","kofta","kogal","kohas","kohen","kohls","koine","koiwi","kojis","kokam","kokas","koker","kokra","kokum","kolas","kolos","kombi","kombu","konbu","kondo","konks","kooks","kooky","koori","kopek","kophs","kopje","koppa","korai","koran","koras","korat","kores","koris","korma","koros","korun","korus","koses","kotch","kotos","kotow","koura","kraal","krabs","kraft","krais","krait","krang","krans","kranz","kraut","krays","kreef","kreen","kreep","kreng","krewe","kriol","krona","krone","kroon","krubi","krump","krunk","ksars","kubie","kudos","kudus","kudzu","kufis","kugel","kuias","kukri","kukus","kulak","kulan","kulas","kulfi","kumis","kumys","kunas","kunds","kuris","kurre","kurta","kurus","kusso","kusti","kutai","kutas","kutch","kutis","kutus","kuyas","kuzus","kvass","kvell","kwaai","kwela","kwink","kwirl","kyack","kyaks","kyang","kyars","kyats","kybos","kydst","kyles","kylie","kylin","kylix","kyloe","kynde","kynds","kypes","kyrie","kytes","kythe","kyudo","laarf","laari","labda","labia","labis","labne","labra","laccy","laced","lacer","laces","lacet","lacey","lacis","lacka","lacks","lacky","laddu","laddy","laded","ladee","lader","lades","ladoo","laers","laevo","lagan","lagar","laggy","lahal","lahar","laich","laics","laide","laids","laigh","laika","laiks","laird","lairs","lairy","laith","laity","laked","laker","lakes","lakhs","lakin","laksa","laldy","lalls","lamas","lambs","lamby","lamed","lamer","lames","lamia","lammy","lamps","lanai","lanas","lanch","lande","lands","laned","lanes","lanks","lants","lapas","lapin","lapis","lapje","lappa","lappy","larch","lards","lardy","laree","lares","larfs","larga","largo","laris","larks","larky","larns","larnt","larum","lased","laser","lases","lassi","lassu","lassy","lasts","latah","lated","laten","latex","lathi","laths","lathy","latke","latus","lauan","lauch","laude","lauds","laufs","laund","laura","laval","lavas","laved","laver","laves","lavra","lavvy","lawed","lawer","lawin","lawks","lawns","lawny","lawsy","laxed","laxer","laxes","laxly","layby","layed","layin","layup","lazar","lazed","lazes","lazos","lazzi","lazzo","leads","leady","leafs","leaks","leams","leans","leany","leaps","leare","lears","leary","leats","leavy","leaze","leben","leccy","leche","ledes","ledgy","ledum","leear","leeks","leeps","leers","leese","leets","leeze","lefte","lefts","leger","leges","legge","leggo","legit","legno","lehrs","lehua","leirs","leish","leman","lemed","lemel","lemes","lemma","lemme","lends","lenes","lengs","lenis","lenos","lense","lenti","lento","leone","lepak","lepid","lepra","lepta","lered","leres","lerps","lesbo","leses","lesos","lests","letch","lethe","letty","letup","leuch","leuco","leuds","leugh","levas","levee","leves","levin","levis","lewis","lexes","lexis","lezes","lezza","lezzo","lezzy","liana","liane","liang","liard","liars","liart","liber","libor","libra","libre","libri","licet","lichi","licht","licit","licks","lidar","lidos","liefs","liens","liers","lieus","lieve","lifer","lifes","lifey","lifts","ligan","liger","ligge","ligne","liked","liker","likes","likin","lills","lilos","lilts","lilty","liman","limas","limax","limba","limbi","limbs","limby","limed","limen","limes","limey","limma","limns","limos","limpa","limps","linac","linch","linds","lindy","lined","lines","liney","linga","lings","lingy","linin","links","linky","linns","linny","linos","lints","linty","linum","linux","lions","lipas","lipes","lipin","lipos","lippy","liras","lirks","lirot","lises","lisks","lisle","lisps","lists","litai","litas","lited","litem","liter","lites","litho","liths","litie","litre","lived","liven","lives","livor","livre","liwaa","liwas","llano","loach","loads","loafs","loams","loans","loast","loave","lobar","lobed","lobes","lobos","lobus","loche","lochs","lochy","locie","locis","locks","locky","locos","locum","loden","lodes","loess","lofts","logan","loges","loggy","logia","logie","logoi","logon","logos","lohan","loids","loins","loipe","loirs","lokes","lokey","lokum","lolas","loled","lollo","lolls","lolly","lolog","lolos","lomas","lomed","lomes","loner","longa","longe","longs","looby","looed","looey","loofa","loofs","looie","looks","looky","looms","loons","loony","loops","loord","loots","loped","loper","lopes","loppy","loral","loran","lords","lordy","lorel","lores","loric","loris","losed","losel","losen","loses","lossy","lotah","lotas","lotes","lotic","lotos","lotsa","lotta","lotte","lotto","lotus","loued","lough","louie","louis","louma","lound","louns","loupe","loups","loure","lours","loury","louts","lovat","loved","lovee","loves","lovey","lovie","lowan","lowed","lowen","lowes","lownd","lowne","lowns","lowps","lowry","lowse","lowth","lowts","loxed","loxes","lozen","luach","luaus","lubed","lubes","lubra","luces","lucks","lucre","ludes","ludic","ludos","luffa","luffs","luged","luger","luges","lulls","lulus","lumas","lumbi","lumme","lummy","lumps","lunas","lunes","lunet","lungi","lungs","lunks","lunts","lupin","lured","lurer","lures","lurex","lurgi","lurgy","lurks","lurry","lurve","luser","lushy","lusks","lusts","lusus","lutea","luted","luter","lutes","luvvy","luxed","luxer","luxes","lweis","lyams","lyard","lyart","lyase","lycea","lycee","lycra","lymes","lynch","lynes","lyres","lysed","lyses","lysin","lysis","lysol","lyssa","lyted","lytes","lythe","lytic","lytta","maaed","maare","maars","maban","mabes","macas","macca","maced","macer","maces","mache","machi","machs","macka","macks","macle","macon","macte","madal","madar","maddy","madge","madid","mados","madre","maedi","maerl","mafic","mafts","magas","mages","maggs","magna","magot","magus","mahal","mahem","mahis","mahoe","mahrs","mahua","mahwa","maids","maiko","maiks","maile","maill","mailo","mails","maims","mains","maire","mairs","maise","maist","majas","majat","majoe","majos","makaf","makai","makan","makar","makee","makes","makie","makis","makos","malae","malai","malam","malar","malas","malax","maleo","males","malic","malik","malis","malky","malls","malms","malmy","malts","malty","malus","malva","malwa","mamak","mamas","mamba","mambu","mamee","mamey","mamie","mamil","manas","manat","mandi","mands","mandy","maneb","maned","maneh","manes","manet","mangi","mangs","manie","manis","manks","manky","manna","manny","manoa","manos","manse","manso","manta","mante","manto","mants","manty","manul","manus","manzo","mapau","mapes","mapou","mappy","maqam","maqui","marae","marah","maral","maran","maras","maray","marcs","mards","mardy","mares","marga","marge","margo","margs","maria","marid","maril","marka","marks","marle","marls","marly","marma","marms","maron","maror","marra","marri","marse","marts","marua","marvy","masas","mased","maser","mases","masha","mashy","masks","massa","massy","masts","masty","masur","masus","masut","matai","mated","mater","mates","mathe","maths","matin","matlo","matra","matsu","matte","matts","matty","matza","matzo","mauby","mauds","mauka","maula","mauls","maums","maumy","maund","maunt","mauri","mausy","mauts","mauvy","mauzy","maven","mavie","mavin","mavis","mawed","mawks","mawky","mawla","mawns","mawps","mawrs","maxed","maxes","maxis","mayan","mayas","mayed","mayos","mayst","mazac","mazak","mazar","mazas","mazed","mazel","mazer","mazes","mazet","mazey","mazut","mbari","mbars","mbila","mbira","mbret","mbube","mbuga","meads","meake","meaks","meals","meane","means","meany","meare","mease","meath","meats","mebbe","mebos","mecha","mechs","mecks","mecum","medii","medin","medle","meech","meeds","meeja","meeps","meers","meets","meffs","meids","meiko","meils","meins","meint","meiny","meism","meith","mekka","melam","melas","melba","melch","melds","meles","melic","melik","mells","meloe","melos","melts","melty","memes","memic","memos","menad","mence","mends","mened","menes","menge","mengs","menil","mensa","mense","mensh","menta","mento","ments","menus","meous","meows","merch","mercs","merde","merds","mered","merel","merer","meres","meril","meris","merks","merle","merls","merse","mersk","mesad","mesal","mesas","mesca","mesel","mesem","meses","meshy","mesia","mesic","mesne","meson","messy","mesto","mesyl","metas","meted","meteg","metel","metes","methi","metho","meths","methy","metic","metif","metis","metol","metre","metta","meums","meuse","meved","meves","mewed","mewls","meynt","mezes","mezza","mezze","mezzo","mgals","mhorr","miais","miaou","miaow","miasm","miaul","micas","miche","michi","micht","micks","micky","micos","micra","middy","midgy","midis","miens","mieux","mieve","miffs","miffy","mifty","miggs","migma","migod","mihas","mihis","mikan","miked","mikes","mikos","mikra","mikva","milch","milds","miler","miles","milfs","milia","milko","milks","mille","mills","milly","milor","milos","milpa","milts","milty","miltz","mimed","mimeo","mimer","mimes","mimis","mimsy","minae","minar","minas","mincy","mindi","minds","mined","mines","minge","mingi","mings","mingy","minis","minke","minks","minny","minos","minse","mints","minxy","miraa","mirah","mirch","mired","mires","mirex","mirid","mirin","mirkn","mirks","mirky","mirls","mirly","miros","mirrl","mirrs","mirvs","mirza","misal","misch","misdo","mises","misgo","misky","misls","misos","missa","misto","mists","misty","mitas","mitch","miter","mites","mitey","mitie","mitis","mitre","mitry","mitta","mitts","mivey","mivvy","mixed","mixen","mixer","mixes","mixie","mixis","mixte","mixup","miyas","mizen","mizes","mizzy","mmkay","mneme","moais","moaky","moals","moana","moans","moany","moars","moats","mobby","mobed","mobee","mobes","mobey","mobie","moble","mobos","mocap","mochi","mochs","mochy","mocks","mocky","mocos","mocus","moder","modes","modge","modii","modin","modoc","modom","modus","moeni","moers","mofos","mogar","mogas","moggy","mogos","mogra","mogue","mohar","mohel","mohos","mohrs","mohua","mohur","moile","moils","moira","moire","moits","moity","mojos","moker","mokes","mokey","mokis","mokky","mokos","mokus","molal","molas","molds","moled","moler","moles","moley","molie","molla","molle","mollo","molls","molly","moloi","molos","molto","molts","molue","molvi","molys","momes","momie","momma","momme","mommy","momos","mompe","momus","monad","monal","monas","monde","mondo","moner","mongo","mongs","monic","monie","monks","monos","monpe","monte","monty","moobs","mooch","moods","mooed","mooey","mooks","moola","mooli","mools","mooly","moong","mooni","moons","moony","moops","moors","moory","mooth","moots","moove","moped","moper","mopes","mopey","moppy","mopsy","mopus","morae","morah","moran","moras","morat","moray","moree","morel","mores","morgy","moria","morin","mormo","morna","morne","morns","moror","morra","morro","morse","morts","moruk","mosed","moses","mosey","mosks","mosso","moste","mosto","mosts","moted","moten","motes","motet","motey","moths","mothy","motis","moton","motte","motts","motty","motus","motza","mouch","moues","moufs","mould","moule","mouls","mouly","moups","moust","mousy","moved","moves","mowas","mowed","mowie","mowra","moxas","moxie","moyas","moyle","moyls","mozed","mozes","mozos","mpret","mrads","msasa","mtepe","mucho","mucic","mucid","mucin","mucko","mucks","mucor","mucro","mudar","mudge","mudif","mudim","mudir","mudra","muffs","muffy","mufti","mugga","muggs","muggy","mugho","mugil","mugos","muhly","muids","muils","muirs","muiry","muist","mujik","mukim","mukti","mulai","mulct","muled","mules","muley","mulga","mulie","mulla","mulls","mulse","mulsh","mumbo","mumms","mumph","mumps","mumsy","mumus","munds","mundu","munga","munge","mungi","mungo","mungs","mungy","munia","munis","munja","munjs","munts","muntu","muons","muras","mured","mures","murex","murgh","murgi","murid","murks","murls","murly","murra","murre","murri","murrs","murry","murth","murti","muruk","murva","musar","musca","mused","musee","muser","muses","muset","musha","musit","musks","musos","musse","mussy","musta","musth","musts","mutas","mutch","muted","muter","mutes","mutha","mutic","mutis","muton","mutti","mutts","mutum","muvva","muxed","muxes","muzak","muzzy","mvula","mvule","mvuli","myall","myals","mylar","mynah","mynas","myoid","myoma","myons","myope","myops","myopy","mysid","mysie","mythi","myths","mythy","myxos","mzees","naams","naans","naats","nabam","nabby","nabes","nabis","nabks","nabla","nabob","nache","nacho","nacre","nadas","naeve","naevi","naffs","nagar","nagas","nages","naggy","nagor","nahal","naiad","naibs","naice","naids","naieo","naifs","naiks","nails","naily","nains","naios","naira","nairu","najib","nakas","naked","naker","nakfa","nalas","naled","nalla","namad","namak","namaz","named","namer","names","namma","namus","nanas","nance","nancy","nandu","nanna","nanos","nante","nanti","nanto","nants","nanty","nanua","napas","naped","napes","napoh","napoo","nappa","nappe","nappy","naras","narco","narcs","nards","nares","naric","naris","narks","narky","narod","narra","narre","nashi","nasho","nasis","nason","nasus","natak","natch","nates","natis","natto","natty","natya","nauch","naunt","navar","naved","naves","navew","navvy","nawab","nawal","nazar","nazes","nazir","nazis","nazzy","nduja","neafe","neals","neant","neaps","nears","neath","neato","neats","nebby","nebek","nebel","neche","necks","neddy","neebs","needs","neefs","neeld","neele","neemb","neems","neeps","neese","neeze","nefie","negri","negro","negus","neifs","neist","neive","nelia","nelis","nelly","nemas","nemic","nemns","nempt","nenes","nenta","neons","neosa","neoza","neper","nepit","neral","neram","nerds","nerfs","nerka","nerks","nerol","nerts","nertz","nervy","neski","nests","nesty","netas","netes","netop","netta","netts","netty","neuks","neume","neums","nevel","neves","nevis","nevus","nevvy","newbs","newed","newel","newie","newsy","newts","nexal","nexin","nexts","nexum","nexus","ngaio","ngaka","ngana","ngapi","ngati","ngege","ngoma","ngoni","ngram","ngwee","nibby","nicad","niced","nicey","nicht","nicks","nicky","nicol","nidal","nided","nides","nidor","nidus","niefs","niess","nieve","nifes","niffs","niffy","nifle","nifty","niger","nigga","nighs","nigre","nigua","nihil","nikab","nikah","nikau","nilas","nills","nimbi","nimbs","nimby","nimps","niner","nines","ninon","ninta","niopo","nioza","nipas","nipet","nippy","niqab","nirls","nirly","nisei","nisin","nisse","nisus","nital","niter","nites","nitid","niton","nitre","nitro","nitry","nitta","nitto","nitty","nival","nivas","nivel","nixed","nixer","nixes","nixie","nizam","njirl","nkosi","nmoli","nmols","noahs","nobby","nocks","nodal","noddy","noded","nodes","nodum","nodus","noels","noema","noeme","nogal","noggs","noggy","nohow","noias","noils","noily","noint","noire","noirs","nokes","noles","nolle","nolls","nolos","nomas","nomen","nomes","nomic","nomoi","nomos","nonan","nonas","nonce","noncy","nonda","nondo","nones","nonet","nongs","nonic","nonis","nonna","nonno","nonny","nonyl","noobs","noois","nooit","nooks","nooky","noone","noons","noops","noove","nopal","noria","norie","noris","norks","norma","norms","nosed","noser","noses","noshi","nosir","notal","notam","noted","noter","notes","notum","nougs","nouja","nould","noule","nouls","nouns","nouny","noups","noust","novae","novas","novia","novio","novum","noway","nowds","nowed","nowls","nowts","nowty","noxal","noxas","noxes","noyau","noyed","noyes","nrtta","nrtya","nsima","nubby","nubia","nucha","nucin","nuddy","nuder","nudes","nudgy","nudie","nudzh","nuevo","nuffs","nugae","nujol","nuked","nukes","nulla","nullo","nulls","nully","numbs","numen","nummy","numps","nunks","nunky","nunny","nunus","nuque","nurds","nurdy","nurls","nurrs","nurts","nurtz","nused","nuses","nutso","nutsy","nyaff","nyala","nyams","nying","nyong","nyssa","nyung","nyuse","nyuze","oafos","oaked","oaker","oakum","oared","oarer","oasal","oases","oasis","oasts","oaten","oater","oaths","oaves","obang","obbos","obeah","obeli","obeys","obias","obied","obiit","obits","objet","oboes","obole","oboli","obols","occam","ocher","oches","ochre","ochry","ocker","ocote","ocrea","octad","octan","octas","octic","octli","octyl","oculi","odahs","odals","odeon","odeum","odism","odist","odium","odoom","odors","odour","odums","odyle","odyls","ofays","offed","offie","oflag","ofter","ofuro","ogams","ogeed","ogees","oggin","ogham","ogive","ogled","ogler","ogles","ogmic","ogres","ohelo","ohias","ohing","ohmic","ohone","oicks","oidia","oiled","oiler","oilet","oinks","oints","oiran","ojime","okapi","okays","okehs","okies","oking","okole","okras","okrug","oktas","olate","oldie","oldly","olehs","oleic","olein","olent","oleos","oleum","oleyl","oligo","olios","oliva","ollas","ollav","oller","ollie","ology","olona","olpae","olpes","omasa","omber","ombus","omdah","omdas","omdda","omdeh","omees","omens","omers","omiai","omits","omlah","ommel","ommin","omnes","omovs","omrah","omuls","oncer","onces","oncet","oncus","ondes","ondol","onely","oners","onery","ongon","onium","onkus","onlap","onlay","onmun","onned","onsen","ontal","ontic","ooaas","oobit","oohed","ooids","oojah","oomph","oonts","oopak","ooped","oopsy","oorie","ooses","ootid","ooyah","oozed","oozes","oozie","oozle","opahs","opals","opens","opepe","opery","opgaf","opihi","oping","oppos","opsat","opsin","opsit","opted","opter","opzit","orach","oracy","orals","orang","orans","orant","orate","orbat","orbed","orbic","orcas","orcin","ordie","ordos","oread","orfes","orful","orgia","orgic","orgue","oribi","oriel","origo","orixa","orles","orlon","orlop","ormer","ornee","ornis","orped","orpin","orris","ortet","ortho","orval","orzos","osars","oscar","osetr","oseys","oshac","osier","oskin","oslin","osmic","osmol","osone","ossia","ostia","otaku","otary","othyl","otium","ottar","ottos","oubit","ouche","oucht","oueds","ouens","ouija","oulks","oumas","oundy","oupas","ouped","ouphe","ouphs","ourey","ourie","ousel","ousia","ousts","outby","outed","outen","outie","outre","outro","outta","ouzel","ouzos","ovals","ovels","ovens","overs","ovism","ovist","ovoli","ovolo","ovule","oware","owari","owche","owers","owies","owled","owler","owlet","owned","ownio","owres","owrie","owsen","oxbow","oxeas","oxers","oxeye","oxids","oxies","oxime","oxims","oxine","oxlip","oxman","oxmen","oxter","oyama","oyers","ozeki","ozena","ozzie","paaho","paals","paans","pacai","pacas","pacay","paced","pacer","paces","pacey","pacha","packs","packy","pacos","pacta","pacts","padam","padas","paddo","padis","padle","padma","padou","padre","padri","paean","paedo","paeon","paged","pager","pages","pagle","pagne","pagod","pagri","pahit","pahos","pahus","paiks","pails","pains","paipe","paips","paire","pairs","paisa","paise","pakay","pakka","pakki","pakua","pakul","palak","palar","palas","palay","palea","paled","pales","palet","palis","palki","palla","palls","pallu","pally","palms","palmy","palpi","palps","palsa","palus","pamby","pampa","panax","pance","panch","panda","pands","pandy","paned","panes","panga","pangs","panim","panir","panko","panks","panna","panne","panni","panny","panto","pants","panty","paoli","paolo","papad","papas","papaw","papes","papey","pappi","pappy","papri","parae","paras","parch","parcs","pardi","pards","pardy","pared","paren","pareo","pares","pareu","parev","parge","pargo","parid","paris","parki","parks","parky","parle","parly","parma","parmo","parms","parol","parps","parra","parrs","parte","parti","parts","parve","parvo","pasag","pasar","pasch","paseo","pases","pasha","pashm","paska","pasmo","paspy","passe","passu","pasts","patas","pated","patee","patel","paten","pater","pates","paths","patia","patin","patka","patly","patta","patte","pattu","patus","pauas","pauls","pauxi","pavan","pavas","paved","paven","paver","paves","pavid","pavie","pavin","pavis","pavon","pavvy","pawas","pawaw","pawed","pawer","pawks","pawky","pawls","pawns","paxes","payed","payor","paysd","peage","peags","peake","peaks","peaky","peals","peans","peare","pears","peart","pease","peasy","peats","peaty","peavy","peaze","pebas","pechs","pecia","pecke","pecks","pecky","pects","pedes","pedis","pedon","pedos","pedro","peece","peeks","peeky","peels","peely","peens","peent","peeoy","peepe","peeps","peepy","peers","peery","peeve","peevo","peggy","peghs","pegma","pegos","peine","peins","peise","peisy","peize","pekan","pekau","pekea","pekes","pekid","pekin","pekoe","pelas","pelau","pelch","peles","pelfs","pells","pelma","pelog","pelon","pelsh","pelta","pelts","pelus","pends","pendu","pened","penes","pengo","penie","penis","penks","penna","penni","pense","pensy","pents","peola","peons","peony","pepla","peple","pepon","pepos","peppy","pepsi","pequi","perae","perai","perce","percs","perdu","perdy","perea","peres","perfs","peris","perks","perle","perls","perms","permy","perne","perns","perog","perps","perry","perse","persp","perst","perts","perve","pervo","pervs","pervy","pesch","pesos","pesta","pests","pesty","petar","peter","petit","petos","petre","petri","petti","petto","pewed","pewee","pewit","peyse","pfftt","phage","phang","phare","pharm","phasm","pheer","pheme","phene","pheon","phese","phial","phies","phish","phizz","phlox","phobe","phoca","phono","phons","phooh","phooo","phota","phots","photy","phpht","phubs","phuts","phutu","phwat","phyla","phyle","phyma","phynx","physa","piais","piani","pians","pibal","pical","picas","piccy","picey","pichi","picks","picon","picot","picra","picul","pieds","piend","piers","piert","pieta","piets","piezo","pight","pigly","pigmy","piing","pikas","pikau","piked","pikel","piker","pikes","pikey","pikis","pikul","pilae","pilaf","pilao","pilar","pilau","pilaw","pilch","pilea","piled","pilei","piler","piles","piley","pilin","pilis","pills","pilon","pilow","pilum","pilus","pimas","pimps","pinas","pinax","pince","pinda","pinds","pined","piner","pines","pinga","pinge","pingo","pings","pinko","pinks","pinna","pinny","pinol","pinon","pinot","pinta","pints","pinup","pions","piony","pious","pioye","pioys","pipal","pipas","piped","pipes","pipet","pipid","pipis","pipit","pippy","pipul","piqui","pirai","pirks","pirls","pirns","pirog","pirre","pirri","pirrs","pisco","pises","pisky","pisos","pissy","piste","pitas","piths","piton","pitot","pitso","pitsu","pitta","pittu","piuma","piums","pivos","pixes","piyut","pized","pizer","pizes","plaas","plack","plaga","plage","plaig","planc","planh","plans","plaps","plash","plasm","plast","plats","platt","platy","plaud","plaur","plavs","playa","plays","pleas","plebe","plebs","pleck","pleep","plein","plena","plene","pleno","pleon","plesh","plets","plews","plexi","plica","plies","pligs","plims","pling","plink","plips","plish","ploat","ploce","plock","plods","ploit","plomb","plong","plonk","plook","ploot","plops","plore","plots","plotz","plouk","plout","plows","plowt","ploye","ploys","pluds","plues","pluff","plugs","pluke","plums","plumy","plung","pluot","plups","plute","pluto","pluty","plyer","pneus","poach","poaka","poake","poalo","pobby","poboy","pocan","poche","pocho","pocks","pocky","podal","poddy","podex","podge","podgy","podia","podos","podus","poems","poena","poeps","poete","poets","pogey","pogge","poggy","pogos","pogue","pohed","poilu","poind","poire","pokal","poked","pokes","pokey","pokie","pokit","poled","poler","poles","poley","polio","polis","polje","polks","pollo","polls","polly","polos","polts","polys","pomas","pombe","pomes","pomme","pommy","pomos","pompa","pomps","ponce","poncy","ponds","pondy","pones","poney","ponga","pongo","pongs","pongy","ponks","ponor","ponto","ponts","ponty","ponzu","pooay","poods","pooed","pooey","poofs","poofy","poohs","poohy","pooja","pooka","pooks","pools","pooly","poons","poopa","poops","poopy","poori","poort","poots","pooty","poove","poovy","popes","popia","popos","poppa","popsy","popup","porae","poral","pored","porer","pores","porey","porge","porgy","porin","porks","porky","porno","porns","porny","porta","porte","porth","ports","porty","porus","posca","posed","poses","poset","posey","posho","posol","poste","posts","potae","potai","potch","poted","potes","potin","potoo","potro","potsy","potto","potts","potty","pouce","pouff","poufs","poufy","pouis","pouke","pouks","poule","poulp","poult","poupe","poupt","pours","pousy","pouts","povos","powan","powie","powin","powis","powlt","pownd","powns","powny","powre","powsy","poxed","poxes","poyas","poynt","poyou","poyse","pozzy","praam","prads","prags","prahu","prams","prana","prang","praos","praps","prase","prate","prats","pratt","praty","praus","prays","preak","predy","preed","preem","prees","preif","preke","prems","premy","prent","preon","preop","preps","presa","prese","prest","preta","preux","preve","prexy","preys","prial","prian","pricy","pridy","prief","prier","pries","prigs","prill","prima","primi","primp","prims","primy","pring","prink","prion","prise","priss","prius","proal","proas","probs","proby","prodd","prods","proem","profs","progs","proin","proke","prole","proll","promo","proms","pronk","prook","proot","props","prora","prore","proso","pross","prost","prosy","proto","proul","prowk","prows","proyn","pruno","prunt","pruny","pruta","pryan","pryer","pryse","pseud","pshaw","pshut","psias","psion","psoae","psoai","psoas","psora","psych","psyop","ptish","ptype","pubby","pubco","pubes","pubis","pubsy","pucan","pucer","puces","pucka","pucks","puddy","pudge","pudic","pudor","pudsy","pudus","puers","puffa","puffs","puggy","pugil","puhas","pujah","pujas","pukas","puked","puker","pukes","pukey","pukka","pukus","pulao","pulas","puled","puler","pules","pulik","pulis","pulka","pulks","pulli","pulls","pully","pulmo","pulps","pulus","pulut","pumas","pumie","pumps","pumpy","punas","punce","punga","pungi","pungo","pungs","pungy","punim","punji","punka","punks","punky","punny","punto","punts","punty","pupae","pupal","pupas","puppa","pupus","purao","purau","purda","purdy","pured","pures","purga","purin","puris","purls","puros","purps","purpy","purre","purrs","purry","pursy","purty","puses","pusle","pussy","putas","puter","putid","putin","puton","putos","putti","putto","putts","puttu","putza","puuko","puyas","puzel","puzta","pwned","pyats","pyets","pygal","pyins","pylon","pyned","pynes","pyoid","pyots","pyral","pyran","pyres","pyrex","pyric","pyros","pyrus","pyuff","pyxed","pyxes","pyxie","pyxis","pzazz","qadis","qaids","qajaq","qanat","qapik","qibla","qilas","qipao","qophs","qorma","quabs","quads","quaff","quags","quair","quais","quaky","quale","qualy","quank","quant","quare","quarl","quass","quate","quats","quawk","quaws","quayd","quays","qubit","quean","queck","queek","queem","queme","quena","quern","queso","quete","queyn","queys","queyu","quibs","quich","quids","quies","quiff","quila","quims","quina","quine","quink","quino","quins","quint","quipo","quips","quipu","quire","quirl","quirt","quist","quits","quoad","quods","quoif","quoin","quois","quoit","quoll","quonk","quops","quork","quorl","quouk","quoys","quran","qursh","quyte","raads","raake","rabat","rabic","rabis","raced","races","rache","racks","racon","raddi","raddy","radge","radgy","radif","radix","radon","rafee","raffs","raffy","rafik","rafiq","rafts","rafty","ragas","ragde","raged","ragee","rager","rages","ragga","raggs","raggy","ragis","ragus","rahed","rahui","raiah","raias","raids","raike","raiks","raile","rails","raine","rains","raird","raita","raith","raits","rajas","rajes","raked","rakee","raker","rakes","rakhi","rakia","rakis","rakki","raksi","rakus","rales","ralli","ramal","ramee","rames","ramet","ramie","ramin","ramis","rammy","ramon","ramps","ramse","ramsh","ramus","ranas","rance","rando","rands","raned","ranee","ranes","ranga","rangi","rangs","rangy","ranid","ranis","ranke","ranks","ranns","ranny","ranse","rants","ranty","raped","rapee","raper","rapes","raphe","rapin","rappe","rapso","rared","raree","rares","rarks","rasam","rasas","rased","raser","rases","rasps","rasse","rasta","ratal","ratan","ratas","ratch","rated","ratel","rater","rates","ratha","rathe","raths","ratoo","ratos","ratti","ratus","rauli","rauns","raupo","raved","ravel","raver","raves","ravey","ravin","rawdy","rawer","rawin","rawks","rawly","rawns","raxed","raxes","rayah","rayas","rayed","rayle","rayls","rayne","razai","razed","razee","razer","razes","razet","razoo","readd","reads","reais","reaks","realo","reals","reame","reams","reamy","reans","reaps","reard","rears","reast","reata","reate","reave","rebab","rebbe","rebec","rebid","rebit","rebop","rebud","rebuy","recal","recce","recco","reccy","recep","recit","recks","recon","recta","recte","recti","recto","recue","redan","redds","reddy","reded","redes","redia","redid","redif","redig","redip","redly","redon","redos","redox","redry","redub","redug","redux","redye","reeaf","reech","reede","reeds","reefs","reefy","reeks","reeky","reels","reely","reems","reens","reerd","reest","reeve","reeze","refan","refed","refel","reffo","refis","refix","refly","refry","regar","reges","reget","regex","reggo","regia","regie","regle","regma","regna","regos","regot","regur","rehem","reifs","reify","reiki","reiks","reine","reing","reink","reins","reird","reist","reive","rejas","rejig","rejon","reked","rekes","rekey","relet","relie","relit","rello","relos","reman","remap","remen","remet","remex","remix","remou","renay","rends","rendu","reney","renga","rengs","renig","renin","renks","renne","renos","rente","rents","reoil","reorg","repas","repat","repeg","repen","repin","repla","repos","repot","repps","repro","repun","reput","reran","rerig","resam","resat","resaw","resay","resee","reses","resew","resid","resit","resod","resol","resow","resto","rests","resty","resue","resus","retag","retam","retax","retem","retia","retie","retin","retip","retox","reune","reups","revet","revie","revow","rewan","rewax","rewed","rewet","rewin","rewon","rewth","rexes","rezes","rhabd","rheas","rheid","rheme","rheum","rhies","rhime","rhine","rhody","rhomb","rhone","rhumb","rhymy","rhyne","rhyta","riads","rials","riant","riata","riato","ribas","ribby","ribes","riced","ricer","rices","ricey","riche","richt","ricin","ricks","rides","ridgy","ridic","riels","riems","rieve","rifer","riffs","riffy","rifte","rifts","rifty","riggs","rigmo","rigol","rikka","rikwa","riled","riles","riley","rille","rills","rilly","rimae","rimed","rimer","rimes","rimon","rimus","rince","rinds","rindy","rines","ringe","rings","ringy","rinks","rioja","rione","riots","rioty","riped","ripes","ripps","riqqs","rises","rishi","risks","risps","rists","risus","rites","rithe","ritts","ritzy","rivas","rived","rivel","riven","rives","riyal","rizas","roads","roady","roake","roaky","roams","roans","roany","roars","roary","roate","robbo","robed","rober","robes","roble","robug","robur","roche","rocks","roded","rodes","rodny","roers","rogan","roguy","rohan","rohes","rohun","rohus","roids","roils","roily","roins","roist","rojak","rojis","roked","roker","rokes","rokey","rokos","rolag","roleo","roles","rolfs","rolls","rolly","romal","roman","romeo","romer","romps","rompu","rompy","ronde","rondo","roneo","rones","ronin","ronne","ronte","ronts","ronuk","roods","roofs","roofy","rooks","rooky","rooms","roons","roops","roopy","roosa","roose","roots","rooty","roped","roper","ropes","ropey","roque","roral","rores","roric","rorid","rorie","rorts","rorty","rosal","rosco","rosed","roses","roset","rosha","roshi","rosin","rosit","rosps","rossa","rosso","rosti","rosts","rotal","rotan","rotas","rotch","roted","rotes","rotis","rotls","roton","rotos","rotta","rotte","rotto","rotty","rouen","roues","rouet","roufs","rougy","rouks","rouky","roule","rouls","roums","roups","roupy","roust","routh","routs","roved","roven","roves","rowan","rowed","rowel","rowen","rowet","rowie","rowme","rownd","rowns","rowth","rowts","royet","royne","royst","rozes","rozet","rozit","ruach","ruana","rubai","ruban","rubby","rubel","rubes","rubin","rubio","ruble","rubli","rubor","rubus","ruche","ruchy","rucks","rudas","rudds","rudes","rudie","rudis","rueda","ruers","ruffe","ruffs","ruffy","rufus","rugae","rugal","rugas","ruggy","ruice","ruing","ruins","rukhs","ruled","rules","rully","rumal","rumbo","rumen","rumes","rumly","rummy","rumpo","rumps","rumpy","runce","runch","runds","runed","runer","runes","rungs","runic","runny","runos","runts","runty","runup","ruote","rupia","rurps","rurus","rusas","ruses","rushy","rusks","rusky","rusma","russe","rusts","ruths","rutin","rutty","ruvid","ryals","rybat","ryiji","ryijy","ryked","rykes","rymer","rymme","rynds","ryoti","ryots","ryper","rypin","rythe","ryugi","saags","sabal","sabed","saber","sabes","sabha","sabin","sabir","sabji","sable","sabos","sabot","sabra","sabre","sabzi","sacks","sacra","sacre","saddo","saddy","sades","sadhe","sadhu","sadic","sadis","sados","sadza","saeta","safed","safes","sagar","sagas","sager","sages","saggy","sagos","sagum","sahab","saheb","sahib","saice","saick","saics","saids","saiga","sails","saims","saine","sains","sairs","saist","saith","sajou","sakai","saker","sakes","sakia","sakis","sakti","salal","salas","salat","salep","sales","salet","salic","salis","salix","salle","salmi","salol","salop","salpa","salps","salse","salto","salts","salud","salue","salut","saman","samas","samba","sambo","samek","samel","samen","sames","samey","samfi","samfu","sammy","sampi","samps","sanad","sands","saned","sanes","sanga","sangh","sango","sangs","sanko","sansa","santo","sants","saola","sapan","sapid","sapor","saran","sards","sared","saree","sarge","sargo","sarin","sarir","saris","sarks","sarky","sarod","saros","sarus","sarvo","saser","sasin","sasse","satai","satay","sated","satem","sater","sates","satis","sauba","sauch","saugh","sauls","sault","saunf","saunt","saury","sauts","sauve","saved","saver","saves","savey","savin","sawah","sawed","sawer","saxes","sayas","sayed","sayee","sayer","sayid","sayne","sayon","sayst","sazes","scabs","scads","scaff","scags","scail","scala","scall","scams","scand","scans","scapa","scape","scapi","scarp","scars","scart","scath","scats","scatt","scaud","scaup","scaur","scaws","sceat","scena","scend","schav","schif","schmo","schul","schwa","scifi","scind","scire","sclim","scobe","scody","scogs","scoog","scoot","scopa","scops","scorp","scote","scots","scoug","scoup","scowp","scows","scrab","scrae","scrag","scran","scrat","scraw","scray","scrim","scrip","scrob","scrod","scrog","scroo","scrow","scudi","scudo","scuds","scuff","scuft","scugs","sculk","scull","sculp","sculs","scums","scups","scurf","scurs","scuse","scuta","scute","scuts","scuzz","scyes","sdayn","sdein","seals","seame","seams","seamy","seans","seare","sears","sease","seats","seaze","sebum","secco","sechs","sects","seder","sedes","sedge","sedgy","sedum","seeds","seeks","seeld","seels","seely","seems","seeps","seepy","seers","sefer","segar","segas","segni","segno","segol","segos","sehri","seifs","seils","seine","seirs","seise","seism","seity","seiza","sekos","sekts","selah","seles","selfs","selfy","selky","sella","selle","sells","selva","semas","semee","semes","semie","semis","senas","sends","senes","senex","sengi","senna","senor","sensa","sensi","sensu","sente","senti","sents","senvy","senza","sepad","sepal","sepic","sepoy","seppo","septa","septs","serac","serai","seral","sered","serer","seres","serfs","serge","seria","seric","serin","serir","serks","seron","serow","serra","serre","serrs","serry","servo","sesey","sessa","setae","setal","seter","seths","seton","setts","sevak","sevir","sewan","sewar","sewed","sewel","sewen","sewin","sexed","sexer","sexes","sexor","sexto","sexts","seyen","sezes","shads","shags","shahs","shaka","shako","shakt","shalm","shaly","shama","shams","shand","shans","shaps","sharn","shart","shash","shaul","shawm","shawn","shaws","shaya","shays","shchi","sheaf","sheal","sheas","sheds","sheel","shend","sheng","shent","sheol","sherd","shere","shero","shets","sheva","shewn","shews","shiai","shiel","shier","shies","shill","shily","shims","shins","shiok","ships","shirr","shirs","shish","shiso","shist","shite","shits","shiur","shiva","shive","shivs","shlep","shlub","shmek","shmoe","shoat","shoed","shoer","shoes","shogi","shogs","shoji","shojo","shola","shonk","shool","shoon","shoos","shope","shops","shorl","shote","shots","shott","shoud","showd","shows","shoyu","shred","shris","shrow","shtar","shtik","shtum","shtup","shuba","shule","shuln","shuls","shuns","shura","shute","shuts","shwas","shyer","sials","sibbs","sibia","sibyl","sices","sicht","sicko","sicks","sicky","sidas","sided","sider","sides","sidey","sidha","sidhe","sidle","sield","siens","sient","sieth","sieur","sifts","sighs","sigil","sigla","signa","signs","sigri","sijos","sikas","siker","sikes","silds","siled","silen","siler","siles","silex","silks","sills","silos","silts","silty","silva","simar","simas","simba","simis","simps","simul","sinds","sined","sines","sings","sinhs","sinks","sinky","sinsi","sinus","siped","sipes","sippy","sired","siree","sires","sirih","siris","siroc","sirra","sirup","sisal","sises","sista","sists","sitar","sitch","sited","sites","sithe","sitka","situp","situs","siver","sixer","sixes","sixmo","sixte","sizar","sized","sizel","sizer","sizes","skags","skail","skald","skank","skarn","skart","skats","skatt","skaws","skean","skear","skeds","skeed","skeef","skeen","skeer","skees","skeet","skeev","skeez","skegg","skegs","skein","skelf","skell","skelm","skelp","skene","skens","skeos","skeps","skerm","skers","skets","skews","skids","skied","skies","skiey","skimo","skims","skink","skins","skint","skios","skips","skirl","skirr","skite","skits","skive","skivy","sklim","skoal","skobe","skody","skoff","skofs","skogs","skols","skool","skort","skosh","skran","skrik","skroo","skuas","skugs","skyed","skyer","skyey","skyfs","skyre","skyrs","skyte","slabs","slade","slaes","slags","slaid","slake","slams","slane","slank","slaps","slart","slats","slaty","slave","slaws","slays","slebs","sleds","sleer","slews","sleys","slier","slily","slims","slipe","slips","slipt","slish","slits","slive","sloan","slobs","sloes","slogs","sloid","slojd","sloka","slomo","sloom","sloot","slops","slopy","slorm","slots","slove","slows","sloyd","slubb","slubs","slued","slues","sluff","slugs","sluit","slums","slurb","slurs","sluse","sluts","slyer","slype","smaak","smaik","smalm","smalt","smarm","smaze","smeek","smees","smeik","smeke","smerk","smews","smick","smily","smirr","smirs","smits","smize","smogs","smoko","smolt","smoor","smoot","smore","smorg","smout","smowt","smugs","smurs","smush","smuts","snabs","snafu","snags","snaps","snarf","snark","snars","snary","snash","snath","snaws","snead","sneap","snebs","sneck","sneds","sneed","snees","snell","snibs","snick","snied","snies","snift","snigs","snips","snipy","snirt","snits","snive","snobs","snods","snoek","snoep","snogs","snoke","snood","snook","snool","snoot","snots","snowk","snows","snubs","snugs","snush","snyes","soaks","soaps","soare","soars","soave","sobas","socas","soces","socia","socko","socks","socle","sodas","soddy","sodic","sodom","sofar","sofas","softa","softs","softy","soger","sohur","soils","soily","sojas","sojus","sokah","soken","sokes","sokol","solah","solan","solas","solde","soldi","soldo","solds","soled","solei","soler","soles","solon","solos","solum","solus","soman","somas","sonce","sonde","sones","songo","songs","songy","sonly","sonne","sonny","sonse","sonsy","sooey","sooks","sooky","soole","sools","sooms","soops","soote","soots","sophs","sophy","sopor","soppy","sopra","soral","soras","sorbi","sorbo","sorbs","sorda","sordo","sords","sored","soree","sorel","sorer","sores","sorex","sorgo","sorns","sorra","sorta","sorts","sorus","soths","sotol","sotto","souce","souct","sough","souks","souls","souly","soums","soups","soupy","sours","souse","souts","sowar","sowce","sowed","sowff","sowfs","sowle","sowls","sowms","sownd","sowne","sowps","sowse","sowth","soxes","soyas","soyle","soyuz","sozin","spack","spacy","spado","spads","spaed","spaer","spaes","spags","spahi","spail","spain","spait","spake","spald","spale","spall","spalt","spams","spane","spang","spans","spard","spars","spart","spate","spats","spaul","spawl","spaws","spayd","spays","spaza","spazz","speal","spean","speat","specs","spect","speel","speer","speil","speir","speks","speld","spelk","speos","spesh","spets","speug","spews","spewy","spial","spica","spick","spics","spide","spier","spies","spiff","spifs","spiks","spile","spims","spina","spink","spins","spirt","spiry","spits","spitz","spivs","splay","splog","spode","spods","spoom","spoor","spoot","spork","sposa","sposh","sposo","spots","sprad","sprag","sprat","spred","sprew","sprit","sprod","sprog","sprue","sprug","spuds","spued","spuer","spues","spugs","spule","spume","spumy","spurs","sputa","spyal","spyre","squab","squaw","squee","squeg","squid","squit","squiz","srsly","stabs","stade","stags","stagy","staig","stane","stang","stans","staph","staps","starn","starr","stars","stary","stats","statu","staun","staws","stays","stean","stear","stedd","stede","steds","steek","steem","steen","steez","steik","steil","stela","stele","stell","steme","stems","stend","steno","stens","stent","steps","stept","stere","stets","stews","stewy","steys","stich","stied","sties","stilb","stile","stime","stims","stimy","stipa","stipe","stire","stirk","stirp","stirs","stive","stivy","stoae","stoai","stoas","stoat","stobs","stoep","stogs","stogy","stoit","stoln","stoma","stond","stong","stonk","stonn","stook","stoor","stope","stops","stopt","stoss","stots","stott","stoun","stoup","stour","stown","stowp","stows","strad","strae","strag","strak","strep","strew","stria","strig","strim","strop","strow","stroy","strum","stubs","stucs","stude","studs","stull","stulm","stumm","stums","stuns","stupa","stupe","sture","sturt","stush","styed","styes","styli","stylo","styme","stymy","styre","styte","subah","subak","subas","subby","suber","subha","succi","sucks","sucky","sucre","sudan","sudds","sudor","sudsy","suede","suent","suers","suete","suets","suety","sugan","sughs","sugos","suhur","suids","suint","suits","sujee","sukhs","sukis","sukuk","sulci","sulfa","sulfo","sulks","sulls","sulph","sulus","sumis","summa","sumos","sumph","sumps","sunis","sunks","sunna","sunns","sunts","sunup","suona","suped","supes","supra","surah","sural","suras","surat","surds","sured","sures","surfs","surfy","surgy","surra","sused","suses","susus","sutor","sutra","sutta","swabs","swack","swads","swage","swags","swail","swain","swale","swaly","swamy","swang","swank","swans","swaps","swapt","sward","sware","swarf","swart","swats","swayl","sways","sweal","swede","sweed","sweel","sweer","swees","sweir","swelt","swerf","sweys","swies","swigs","swile","swims","swink","swipe","swire","swiss","swith","swits","swive","swizz","swobs","swole","swoll","swoln","swops","swopt","swots","swoun","sybbe","sybil","syboe","sybow","sycee","syces","sycon","syeds","syens","syker","sykes","sylis","sylph","sylva","symar","synch","syncs","synds","syned","synes","synth","syped","sypes","syphs","syrah","syren","sysop","sythe","syver","taals","taata","tabac","taber","tabes","tabid","tabis","tabla","tabls","tabor","tabos","tabun","tabus","tacan","taces","tacet","tache","tachi","tacho","tachs","tacks","tacos","tacts","tadah","taels","tafia","taggy","tagma","tagua","tahas","tahrs","taiga","taigs","taiko","tails","tains","taira","taish","taits","tajes","takas","takes","takhi","takht","takin","takis","takky","talak","talaq","talar","talas","talcs","talcy","talea","taler","tales","talik","talks","talky","talls","talma","talpa","taluk","talus","tamal","tamas","tamed","tames","tamin","tamis","tammy","tamps","tanas","tanga","tangi","tangs","tanhs","tania","tanka","tanks","tanky","tanna","tansu","tansy","tante","tanti","tanto","tanty","tapas","taped","tapen","tapes","tapet","tapis","tappa","tapus","taras","tardo","tards","tared","tares","targa","targe","tarka","tarns","taroc","tarok","taros","tarps","tarre","tarry","tarse","tarsi","tarte","tarts","tarty","tarzy","tasar","tasca","tased","taser","tases","tasks","tassa","tasse","tasso","tasto","tatar","tater","tates","taths","tatie","tatou","tatts","tatus","taube","tauld","tauon","taupe","tauts","tauty","tavah","tavas","taver","tawaf","tawai","tawas","tawed","tawer","tawie","tawse","tawts","taxed","taxer","taxes","taxis","taxol","taxon","taxor","taxus","tayra","tazza","tazze","teade","teads","teaed","teaks","teals","teams","tears","teats","teaze","techs","techy","tecta","tecum","teels","teems","teend","teene","teens","teeny","teers","teets","teffs","teggs","tegua","tegus","tehee","tehrs","teiid","teils","teind","teins","tekke","telae","telco","teles","telex","telia","telic","tells","telly","teloi","telos","temed","temes","tempi","temps","tempt","temse","tench","tends","tendu","tenes","tenge","tenia","tenne","tenno","tenny","tenon","tents","tenty","tenue","tepal","tepas","tepoy","terai","teras","terce","terek","teres","terfe","terfs","terga","terms","terne","terns","terre","terry","terts","terza","tesla","testa","teste","tests","tetes","teths","tetra","tetri","teuch","teugh","tewed","tewel","tewit","texas","texes","texta","texts","thack","thagi","thaim","thale","thali","thana","thane","thang","thans","thanx","tharm","thars","thaws","thawt","thawy","thebe","theca","theed","theek","thees","thegn","theic","thein","thelf","thema","thens","theor","theow","therm","thesp","thete","thews","thewy","thigs","thilk","thill","thine","thins","thiol","thirl","thoft","thole","tholi","thoro","thorp","thots","thous","thowl","thrae","thraw","thrid","thrip","throe","thuds","thugs","thuja","thunk","thurl","thuya","thymi","thymy","tians","tiare","tiars","tical","ticca","ticed","tices","tichy","ticks","ticky","tiddy","tided","tides","tiefs","tiers","tiffs","tifos","tifts","tiges","tigon","tikas","tikes","tikia","tikis","tikka","tilak","tiled","tiler","tiles","tills","tilly","tilth","tilts","timbo","timed","times","timon","timps","tinas","tinct","tinds","tinea","tined","tines","tinge","tings","tinks","tinny","tinto","tints","tinty","tipis","tippy","tipup","tired","tires","tirls","tiros","tirrs","tirth","titar","titas","titch","titer","tithi","titin","titir","titis","titre","titty","titup","tiyin","tiyns","tizes","tizzy","toads","toady","toaze","tocks","tocky","tocos","todde","todea","todos","toeas","toffs","toffy","tofts","tofus","togae","togas","toged","toges","togue","tohos","toidy","toile","toils","toing","toise","toits","toity","tokay","toked","toker","tokes","tokos","tolan","tolar","tolas","toled","toles","tolls","tolly","tolts","tolus","tolyl","toman","tombo","tombs","tomen","tomes","tomia","tomin","tomme","tommy","tomos","tomoz","tondi","tondo","toned","toner","tones","toney","tongs","tonka","tonks","tonne","tonus","tools","tooms","toons","toots","toped","topee","topek","toper","topes","tophe","tophi","tophs","topis","topoi","topos","toppy","toque","torah","toran","toras","torcs","tores","toric","torii","toros","torot","torrs","torse","torsi","torsk","torta","torte","torts","tosas","tosed","toses","toshy","tossy","tosyl","toted","toter","totes","totty","touks","touns","tours","touse","tousy","touts","touze","touzy","towai","towed","towie","towno","towns","towny","towse","towsy","towts","towze","towzy","toyed","toyer","toyon","toyos","tozed","tozes","tozie","trabs","trads","trady","traga","tragi","trags","tragu","traik","trams","trank","tranq","trans","trant","trape","trapo","traps","trapt","trass","trats","tratt","trave","trayf","trays","treck","treed","treen","trees","trefa","treif","treks","trema","trems","tress","trest","trets","trews","treyf","treys","triac","tride","trier","tries","trifa","triff","trigo","trigs","trike","trild","trill","trims","trine","trins","triol","trior","trios","trips","tripy","trist","troad","troak","troat","trock","trode","trods","trogs","trois","troke","tromp","trona","tronc","trone","tronk","trons","trooz","tropo","troth","trots","trows","troys","trued","trues","trugo","trugs","trull","tryer","tryke","tryma","tryps","tsade","tsadi","tsars","tsked","tsuba","tsubo","tuans","tuart","tuath","tubae","tubar","tubas","tubby","tubed","tubes","tucks","tufas","tuffe","tuffs","tufts","tufty","tugra","tuile","tuina","tuism","tuktu","tules","tulpa","tulps","tulsi","tumid","tummy","tumps","tumpy","tunas","tunds","tuned","tuner","tunes","tungs","tunny","tupek","tupik","tuple","tuque","turds","turfs","turfy","turks","turme","turms","turns","turnt","turon","turps","turrs","tushy","tusks","tusky","tutee","tutes","tutti","tutty","tutus","tuxes","tuyer","twaes","twain","twals","twank","twats","tways","tweel","tween","tweep","tweer","twerk","twerp","twier","twigs","twill","twilt","twink","twins","twiny","twire","twirk","twirp","twite","twits","twocs","twoer","twonk","twyer","tyees","tyers","tyiyn","tykes","tyler","tymps","tynde","tyned","tynes","typal","typed","types","typey","typic","typos","typps","typto","tyran","tyred","tyres","tyros","tythe","tzars","ubacs","ubity","udals","udons","udyog","ugali","ugged","uhlan","uhuru","ukase","ulama","ulans","ulema","ulmin","ulmos","ulnad","ulnae","ulnar","ulnas","ulpan","ulvas","ulyie","ulzie","umami","umbel","umber","umble","umbos","umbre","umiac","umiak","umiaq","ummah","ummas","ummed","umped","umphs","umpie","umpty","umrah","umras","unagi","unais","unapt","unarm","unary","unaus","unbag","unban","unbar","unbed","unbid","unbox","uncap","unces","uncia","uncos","uncoy","uncus","undam","undee","undos","undug","uneth","unfix","ungag","unget","ungod","ungot","ungum","unhat","unhip","unica","unios","units","unjam","unked","unket","unkey","unkid","unkut","unlap","unlaw","unlay","unled","unleg","unlet","unlid","unmad","unman","unmew","unmix","unode","unold","unown","unpay","unpeg","unpen","unpin","unply","unpot","unput","unred","unrid","unrig","unrip","unsaw","unsay","unsee","unsew","unsex","unsod","unsub","untag","untax","untin","unwet","unwit","unwon","upbow","upbye","updos","updry","upend","upful","upjet","uplay","upled","uplit","upped","upran","uprun","upsee","upsey","uptak","upter","uptie","uraei","urali","uraos","urare","urari","urase","urate","urbex","urbia","urdee","ureal","ureas","uredo","ureic","ureid","urena","urent","urged","urger","urges","urial","urite","urman","urnal","urned","urped","ursae","ursid","urson","urubu","urupa","urvas","usens","users","useta","usnea","usnic","usque","ustad","uster","usure","usury","uteri","utero","uveal","uveas","uvula","vacas","vacay","vacua","vacui","vacuo","vadas","vaded","vades","vadge","vagal","vagus","vaids","vails","vaire","vairs","vairy","vajra","vakas","vakil","vales","valis","valli","valse","vamps","vampy","vanda","vaned","vanes","vanga","vangs","vants","vaped","vaper","vapes","varan","varas","varda","vardo","vardy","varec","vares","varia","varix","varna","varus","varve","vasal","vases","vasts","vasty","vatas","vatha","vatic","vatje","vatos","vatus","vauch","vaute","vauts","vawte","vaxes","veale","veals","vealy","veena","veeps","veers","veery","vegas","veges","veggo","vegie","vegos","vehme","veils","veily","veins","veiny","velar","velds","veldt","veles","vells","velum","venae","venal","venas","vends","vendu","veney","venge","venin","venti","vents","venus","verba","verbs","verde","verra","verre","verry","versa","verst","verte","verts","vertu","vespa","vesta","vests","vetch","veuve","veves","vexed","vexer","vexes","vexil","vezir","vials","viand","vibed","vibes","vibex","vibey","viced","vices","vichy","vicus","viers","vieux","views","viewy","vifda","viffs","vigas","vigia","vilde","viler","ville","villi","vills","vimen","vinal","vinas","vinca","vined","viner","vines","vinew","vinho","vinic","vinny","vinos","vints","viold","viols","vired","vireo","vires","virga","virge","virgo","virid","virls","virtu","visas","vised","vises","visie","visna","visne","vison","visto","vitae","vitas","vitex","vitro","vitta","vivas","vivat","vivda","viver","vives","vivos","vivre","vizir","vizor","vlast","vleis","vlies","vlogs","voars","vobla","vocab","voces","voddy","vodou","vodun","voema","vogie","voici","voids","voile","voips","volae","volar","voled","voles","volet","volke","volks","volta","volte","volti","volts","volva","volve","vomer","voted","votes","vouge","voulu","vowed","vower","voxel","voxes","vozhd","vraic","vrils","vroom","vrous","vrouw","vrows","vuggs","vuggy","vughs","vughy","vulgo","vulns","vulva","vutty","vygie","waacs","wacke","wacko","wacks","wadas","wadds","waddy","waded","wader","wades","wadge","wadis","wadts","waffs","wafts","waged","wages","wagga","wagyu","wahay","wahey","wahoo","waide","waifs","waift","wails","wains","wairs","waite","waits","wakas","waked","waken","waker","wakes","wakfs","waldo","walds","waled","waler","wales","walie","walis","walks","walla","walls","wally","walty","wamed","wames","wamus","wands","waned","wanes","waney","wangs","wanks","wanky","wanle","wanly","wanna","wanta","wants","wanty","wanze","waqfs","warbs","warby","wards","wared","wares","warez","warks","warms","warns","warps","warre","warst","warts","wases","washi","washy","wasms","wasps","waspy","wasts","watap","watts","wauff","waugh","wauks","waulk","wauls","waurs","waved","waves","wavey","wawas","wawes","wawls","waxed","waxer","waxes","wayed","wazir","wazoo","weald","weals","weamb","weans","wears","webby","weber","wecht","wedel","wedgy","weeds","weeis","weeke","weeks","weels","weems","weens","weeny","weeps","weepy","weest","weete","weets","wefte","wefts","weids","weils","weirs","weise","weize","wekas","welds","welke","welks","welkt","wells","welly","welts","wembs","wench","wends","wenge","wenny","wents","werfs","weros","wersh","wests","wetas","wetly","wexed","wexes","whamo","whams","whang","whaps","whare","whata","whats","whaup","whaur","wheal","whear","wheek","wheen","wheep","wheft","whelk","whelm","whens","whets","whews","wheys","whids","whies","whift","whigs","whilk","whims","whins","whios","whips","whipt","whirr","whirs","whish","whiss","whist","whits","whity","whizz","whomp","whoof","whoot","whops","whore","whorl","whort","whoso","whows","whump","whups","whyda","wicca","wicks","wicky","widdy","wides","wiels","wifed","wifes","wifey","wifie","wifts","wifty","wigan","wigga","wiggy","wikis","wilco","wilds","wiled","wiles","wilga","wilis","wilja","wills","wilts","wimps","winds","wined","wines","winey","winge","wings","wingy","winks","winky","winna","winns","winos","winze","wiped","wiper","wipes","wired","wirer","wires","wirra","wirri","wised","wises","wisha","wisht","wisps","wists","witan","wited","wites","withe","withs","withy","wived","wiver","wives","wizen","wizes","wizzo","woads","woady","woald","wocks","wodge","wodgy","woful","wojus","woker","wokka","wolds","wolfs","wolly","wolve","womas","wombs","womby","womyn","wonga","wongi","wonks","wonky","wonts","woods","wooed","woofs","woofy","woold","wools","woons","woops","woopy","woose","woosh","wootz","words","works","worky","worms","wormy","worts","wowed","wowee","wowse","woxen","wrang","wraps","wrapt","wrast","wrate","wrawl","wrens","wrick","wried","wrier","wries","writs","wroke","wroot","wroth","wryer","wuddy","wudus","wuffs","wulls","wunga","wurst","wuses","wushu","wussy","wuxia","wyled","wyles","wynds","wynns","wyted","wytes","wythe","xebec","xenia","xenic","xenon","xeric","xerox","xerus","xoana","xolos","xrays","xviii","xylan","xylem","xylic","xylol","xylyl","xysti","xysts","yaars","yaass","yabas","yabba","yabby","yacca","yacka","yacks","yadda","yaffs","yager","yages","yagis","yagna","yahoo","yaird","yajna","yakka","yakow","yales","yamen","yampa","yampy","yamun","yandy","yangs","yanks","yapok","yapon","yapps","yappy","yarak","yarco","yards","yarer","yarfa","yarks","yarns","yarra","yarrs","yarta","yarto","yates","yatra","yauds","yauld","yaups","yawed","yawey","yawls","yawns","yawny","yawps","yayas","ybore","yclad","ycled","ycond","ydrad","ydred","yeads","yeahs","yealm","yeans","yeard","years","yecch","yechs","yechy","yedes","yeeds","yeeek","yeesh","yeggs","yelks","yells","yelms","yelps","yelts","yenta","yente","yerba","yerds","yerks","yeses","yesks","yests","yesty","yetis","yetts","yeuch","yeuks","yeuky","yeven","yeves","yewen","yexed","yexes","yfere","yiked","yikes","yills","yince","yipes","yippy","yirds","yirks","yirrs","yirth","yites","yitie","ylems","ylide","ylids","ylike","ylkes","ymolt","ympes","yobbo","yobby","yocks","yodel","yodhs","yodle","yogas","yogee","yoghs","yogic","yogin","yogis","yohah","yohay","yoick","yojan","yokan","yoked","yokeg","yokel","yoker","yokes","yokul","yolks","yolky","yolps","yomim","yomps","yonic","yonis","yonks","yonny","yoofs","yoops","yopos","yoppo","yores","yorga","yorks","yorps","youks","yourn","yours","yourt","youse","yowed","yowes","yowie","yowls","yowsa","yowza","yoyos","yrapt","yrent","yrivd","yrneh","ysame","ytost","yuans","yucas","yucca","yucch","yucko","yucks","yucky","yufts","yugas","yuked","yukes","yukky","yukos","yulan","yules","yummo","yummy","yumps","yupon","yuppy","yurta","yurts","yuzus","zabra","zacks","zaida","zaide","zaidy","zaire","zakat","zamac","zamak","zaman","zambo","zamia","zamis","zanja","zante","zanza","zanze","zappy","zarda","zarfs","zaris","zatis","zawns","zaxes","zayde","zayin","zazen","zeals","zebec","zebub","zebus","zedas","zeera","zeins","zendo","zerda","zerks","zeros","zests","zetas","zexes","zezes","zhomo","zhush","zhuzh","zibet","ziffs","zigan","zikrs","zilas","zilch","zilla","zills","zimbi","zimbs","zinco","zincs","zincy","zineb","zines","zings","zingy","zinke","zinky","zinos","zippo","zippy","ziram","zitis","zitty","zizel","zizit","zlote","zloty","zoaea","zobos","zobus","zocco","zoeae","zoeal","zoeas","zoism","zoist","zokor","zolle","zombi","zonae","zonda","zoned","zoner","zones","zonks","zooea","zooey","zooid","zooks","zooms","zoomy","zoons","zooty","zoppa","zoppo","zoril","zoris","zorro","zorse","zouks","zowee","zowie","zulus","zupan","zupas","zuppa","zurfs","zuzim","zygal","zygon","zymes","zymic","cigar","rebut","sissy","humph","awake","blush","focal","evade","naval","serve","heath","dwarf","model","karma","stink","grade","quiet","bench","abate","feign","major","death","fresh","crust","stool","colon","abase","marry","react","batty","pride","floss","helix","croak","staff","paper","unfed","whelp","trawl","outdo","adobe","crazy","sower","repay","digit","crate","cluck","spike","mimic","pound","maxim","linen","unmet","flesh","booby","forth","first","stand","belly","ivory","seedy","print","yearn","drain","bribe","stout","panel","crass","flume","offal","agree","error","swirl","argue","bleed","delta","flick","totem","wooer","front","shrub","parry","biome","lapel","start","greet","goner","golem","lusty","loopy","round","audit","lying","gamma","labor","islet","civic","forge","corny","moult","basic","salad","agate","spicy","spray","essay","fjord","spend","kebab","guild","aback","motor","alone","hatch","hyper","thumb","dowry","ought","belch","dutch","pilot","tweed","comet","jaunt","enema","steed","abyss","growl","fling","dozen","boozy","erode","world","gouge","click","briar","great","altar","pulpy","blurt","coast","duchy","groin","fixer","group","rogue","badly","smart","pithy","gaudy","chill","heron","vodka","finer","surer","radio","rouge","perch","retch","wrote","clock","tilde","store","prove","bring","solve","cheat","grime","exult","usher","epoch","triad","break","rhino","viral","conic","masse","sonic","vital","trace","using","peach","champ","baton","brake","pluck","craze","gripe","weary","picky","acute","ferry","aside","tapir","troll","unify","rebus","boost","truss","siege","tiger","banal","slump","crank","gorge","query","drink","favor","abbey","tangy","panic","solar","shire","proxy","point","robot","prick","wince","crimp","knoll","sugar","whack","mount","perky","could","wrung","light","those","moist","shard","pleat","aloft","skill","elder","frame","humor","pause","ulcer","ultra","robin","cynic","aroma","caulk","shake","dodge","swill","tacit","other","thorn","trove","bloke","vivid","spill","chant","choke","rupee","nasty","mourn","ahead","brine","cloth","hoard","sweet","month","lapse","watch","today","focus","smelt","tease","cater","movie","saute","allow","renew","their","slosh","purge","chest","depot","epoxy","nymph","found","shall","stove","lowly","snout","trope","fewer","shawl","natal","comma","foray","scare","stair","black","squad","royal","chunk","mince","shame","cheek","ample","flair","foyer","cargo","oxide","plant","olive","inert","askew","heist","shown","zesty","trash","larva","forgo","story","hairy","train","homer","badge","midst","canny","shine","gecko","farce","slung","tipsy","metal","yield","delve","being","scour","glass","gamer","scrap","money","hinge","album","vouch","asset","tiara","crept","bayou","atoll","manor","creak","showy","phase","froth","depth","gloom","flood","trait","girth","piety","goose","float","donor","atone","primo","apron","blown","cacao","loser","input","gloat","awful","brink","smite","beady","rusty","retro","droll","gawky","hutch","pinto","egret","lilac","sever","field","fluff","agape","voice","stead","berth","madam","night","bland","liver","wedge","roomy","wacky","flock","angry","trite","aphid","tryst","midge","power","elope","cinch","motto","stomp","upset","bluff","cramp","quart","coyly","youth","rhyme","buggy","alien","smear","unfit","patty","cling","glean","label","hunky","khaki","poker","gruel","twice","twang","shrug","treat","waste","merit","woven","needy","clown","irony","ruder","gauze","chief","onset","prize","fungi","charm","gully","inter","whoop","taunt","leery","class","theme","lofty","tibia","booze","alpha","thyme","doubt","parer","chute","stick","trice","alike","recap","saint","glory","grate","admit","brisk","soggy","usurp","scald","scorn","leave","twine","sting","bough","marsh","sloth","dandy","vigor","howdy","enjoy","valid","ionic","equal","floor","catch","spade","stein","exist","quirk","denim","grove","spiel","mummy","fault","foggy","flout","carry","sneak","libel","waltz","aptly","piney","inept","aloud","photo","dream","stale","unite","snarl","baker","there","glyph","pooch","hippy","spell","folly","louse","gulch","vault","godly","threw","fleet","grave","inane","shock","crave","spite","valve","skimp","claim","rainy","musty","pique","daddy","quasi","arise","aging","valet","opium","avert","stuck","recut","mulch","genre","plume","rifle","count","incur","total","wrest","mocha","deter","study","lover","safer","rivet","funny","smoke","mound","undue","sedan","pagan","swine","guile","gusty","equip","tough","canoe","chaos","covet","human","udder","lunch","blast","stray","manga","melee","lefty","quick","paste","given","octet","risen","groan","leaky","grind","carve","loose","sadly","spilt","apple","slack","honey","final","sheen","eerie","minty","slick","derby","wharf","spelt","coach","erupt","singe","price","spawn","fairy","jiffy","filmy","stack","chose","sleep","ardor","nanny","niece","woozy","handy","grace","ditto","stank","cream","usual","diode","valor","angle","ninja","muddy","chase","reply","prone","spoil","heart","shade","diner","arson","onion","sleet","dowel","couch","palsy","bowel","smile","evoke","creek","lance","eagle","idiot","siren","built","embed","award","dross","annul","goody","frown","patio","laden","humid","elite","lymph","edify","might","reset","visit","gusto","purse","vapor","crock","write","sunny","loath","chaff","slide","queer","venom","stamp","sorry","still","acorn","aping","pushy","tamer","hater","mania","awoke","brawn","swift","exile","birch","lucky","freer","risky","ghost","plier","lunar","winch","snare","nurse","house","borax","nicer","lurch","exalt","about","savvy","toxin","tunic","pried","inlay","chump","lanky","cress","eater","elude","cycle","kitty","boule","moron","tenet","place","lobby","plush","vigil","index","blink","clung","qualm","croup","clink","juicy","stage","decay","nerve","flier","shaft","crook","clean","china","ridge","vowel","gnome","snuck","icing","spiny","rigor","snail","flown","rabid","prose","thank","poppy","budge","fiber","moldy","dowdy","kneel","track","caddy","quell","dumpy","paler","swore","rebar","scuba","splat","flyer","horny","mason","doing","ozone","amply","molar","ovary","beset","queue","cliff","magic","truce","sport","fritz","edict","twirl","verse","llama","eaten","range","whisk","hovel","rehab","macaw","sigma","spout","verve","sushi","dying","fetid","brain","buddy","thump","scion","candy","chord","basin","march","crowd","arbor","gayly","musky","stain","dally","bless","bravo","stung","title","ruler","kiosk","blond","ennui","layer","fluid","tatty","score","cutie","zebra","barge","matey","bluer","aider","shook","river","privy","betel","frisk","bongo","begun","azure","weave","genie","sound","glove","braid","scope","wryly","rover","assay","ocean","bloom","irate","later","woken","silky","wreck","dwelt","slate","smack","solid","amaze","hazel","wrist","jolly","globe","flint","rouse","civil","vista","relax","cover","alive","beech","jetty","bliss","vocal","often","dolly","eight","joker","since","event","ensue","shunt","diver","poser","worst","sweep","alley","creed","anime","leafy","bosom","dunce","stare","pudgy","waive","choir","stood","spoke","outgo","delay","bilge","ideal","clasp","seize","hotly","laugh","sieve","block","meant","grape","noose","hardy","shied","drawl","daisy","putty","strut","burnt","tulip","crick","idyll","vixen","furor","geeky","cough","naive","shoal","stork","bathe","aunty","check","prime","brass","outer","furry","razor","elect","evict","imply","demur","quota","haven","cavil","swear","crump","dough","gavel","wagon","salon","nudge","harem","pitch","sworn","pupil","excel","stony","cabin","unzip","queen","trout","polyp","earth","storm","until","taper","enter","child","adopt","minor","fatty","husky","brave","filet","slime","glint","tread","steal","regal","guest","every","murky","share","spore","hoist","buxom","inner","otter","dimly","level","sumac","donut","stilt","arena","sheet","scrub","fancy","slimy","pearl","silly","porch","dingo","sepia","amble","shady","bread","friar","reign","dairy","quill","cross","brood","tuber","shear","posit","blank","villa","shank","piggy","freak","which","among","fecal","shell","would","algae","large","rabbi","agony","amuse","bushy","copse","swoon","knife","pouch","ascot","plane","crown","urban","snide","relay","abide","viola","rajah","straw","dilly","crash","amass","third","trick","tutor","woody","blurb","grief","disco","where","sassy","beach","sauna","comic","clued","creep","caste","graze","snuff","frock","gonad","drunk","prong","lurid","steel","halve","buyer","vinyl","utile","smell","adage","worry","tasty","local","trade","finch","ashen","modal","gaunt","clove","enact","adorn","roast","speck","sheik","missy","grunt","snoop","party","touch","mafia","emcee","array","south","vapid","jelly","skulk","angst","tubal","lower","crest","sweat","cyber","adore","tardy","swami","notch","groom","roach","hitch","young","align","ready","frond","strap","puree","realm","venue","swarm","offer","seven","dryer","diary","dryly","drank","acrid","heady","theta","junto","pixie","quoth","bonus","shalt","penne","amend","datum","build","piano","shelf","lodge","suing","rearm","coral","ramen","worth","psalm","infer","overt","mayor","ovoid","glide","usage","poise","randy","chuck","prank","fishy","tooth","ether","drove","idler","swath","stint","while","begat","apply","slang","tarot","radar","credo","aware","canon","shift","timer","bylaw","serum","three","steak","iliac","shirk","blunt","puppy","penal","joist","bunny","shape","beget","wheel","adept","stunt","stole","topaz","chore","fluke","afoot","bloat","bully","dense","caper","sneer","boxer","jumbo","lunge","space","avail","short","slurp","loyal","flirt","pizza","conch","tempo","droop","plate","bible","plunk","afoul","savoy","steep","agile","stake","dwell","knave","beard","arose","motif","smash","broil","glare","shove","baggy","mammy","swamp","along","rugby","wager","quack","squat","snaky","debit","mange","skate","ninth","joust","tramp","spurn","medal","micro","rebel","flank","learn","nadir","maple","comfy","remit","gruff","ester","least","mogul","fetch","cause","oaken","aglow","meaty","gaffe","shyly","racer","prowl","thief","stern","poesy","rocky","tweet","waist","spire","grope","havoc","patsy","truly","forty","deity","uncle","swish","giver","preen","bevel","lemur","draft","slope","annoy","lingo","bleak","ditty","curly","cedar","dirge","grown","horde","drool","shuck","crypt","cumin","stock","gravy","locus","wider","breed","quite","chafe","cache","blimp","deign","fiend","logic","cheap","elide","rigid","false","renal","pence","rowdy","shoot","blaze","envoy","posse","brief","never","abort","mouse","mucky","sulky","fiery","media","trunk","yeast","clear","skunk","scalp","bitty","cider","koala","duvet","segue","creme","super","grill","after","owner","ember","reach","nobly","empty","speed","gipsy","recur","smock","dread","merge","burst","kappa","amity","shaky","hover","carol","snort","synod","faint","haunt","flour","chair","detox","shrew","tense","plied","quark","burly","novel","waxen","stoic","jerky","blitz","beefy","lyric","hussy","towel","quilt","below","bingo","wispy","brash","scone","toast","easel","saucy","value","spice","honor","route","sharp","bawdy","radii","skull","phony","issue","lager","swell","urine","gassy","trial","flora","upper","latch","wight","brick","retry","holly","decal","grass","shack","dogma","mover","defer","sober","optic","crier","vying","nomad","flute","hippo","shark","drier","obese","bugle","tawny","chalk","feast","ruddy","pedal","scarf","cruel","bleat","tidal","slush","semen","windy","dusty","sally","igloo","nerdy","jewel","shone","whale","hymen","abuse","fugue","elbow","crumb","pansy","welsh","syrup","terse","suave","gamut","swung","drake","freed","afire","shirt","grout","oddly","tithe","plaid","dummy","broom","blind","torch","enemy","again","tying","pesky","alter","gazer","noble","ethos","bride","extol","decor","hobby","beast","idiom","utter","these","sixth","alarm","erase","elegy","spunk","piper","scaly","scold","hefty","chick","sooty","canal","whiny","slash","quake","joint","swept","prude","heavy","wield","femme","lasso","maize","shale","screw","spree","smoky","whiff","scent","glade","spent","prism","stoke","riper","orbit","cocoa","guilt","humus","shush","table","smirk","wrong","noisy","alert","shiny","elate","resin","whole","hunch","pixel","polar","hotel","sword","cleat","mango","rumba","puffy","filly","billy","leash","clout","dance","ovate","facet","chili","paint","liner","curio","salty","audio","snake","fable","cloak","navel","spurt","pesto","balmy","flash","unwed","early","churn","weedy","stump","lease","witty","wimpy","spoof","saner","blend","salsa","thick","warty","manic","blare","squib","spoon","probe","crepe","knack","force","debut","order","haste","teeth","agent","widen","icily","slice","ingot","clash","juror","blood","abode","throw","unity","pivot","slept","troop","spare","sewer","parse","morph","cacti","tacky","spool","demon","moody","annex","begin","fuzzy","patch","water","lumpy","admin","omega","limit","tabby","macho","aisle","skiff","basis","plank","verge","botch","crawl","lousy","slain","cubic","raise","wrack","guide","foist","cameo","under","actor","revue","fraud","harpy","scoop","climb","refer","olden","clerk","debar","tally","ethic","cairn","tulle","ghoul","hilly","crude","apart","scale","older","plain","sperm","briny","abbot","rerun","quest","crisp","bound","befit","drawn","suite","itchy","cheer","bagel","guess","broad","axiom","chard","caput","leant","harsh","curse","proud","swing","opine","taste","lupus","gumbo","miner","green","chasm","lipid","topic","armor","brush","crane","mural","abled","habit","bossy","maker","dusky","dizzy","lithe","brook","jazzy","fifty","sense","giant","surly","legal","fatal","flunk","began","prune","small","slant","scoff","torus","ninny","covey","viper","taken","moral","vogue","owing","token","entry","booth","voter","chide","elfin","ebony","neigh","minim","melon","kneed","decoy","voila","ankle","arrow","mushy","tribe","cease","eager","birth","graph","odder","terra","weird","tried","clack","color","rough","weigh","uncut","ladle","strip","craft","minus","dicey","titan","lucid","vicar","dress","ditch","gypsy","pasta","taffy","flame","swoop","aloof","sight","broke","teary","chart","sixty","wordy","sheer","leper","nosey","bulge","savor","clamp","funky","foamy","toxic","brand","plumb","dingy","butte","drill","tripe","bicep","tenor","krill","worse","drama","hyena","think","ratio","cobra","basil","scrum","bused","phone","court","camel","proof","heard","angel","petal","pouty","throb","maybe","fetal","sprig","spine","shout","cadet","macro","dodgy","satyr","rarer","binge","trend","nutty","leapt","amiss","split","myrrh","width","sonar","tower","baron","fever","waver","spark","belie","sloop","expel","smote","baler","above","north","wafer","scant","frill","awash","snack","scowl","frail","drift","limbo","fence","motel","ounce","wreak","revel","talon","prior","knelt","cello","flake","debug","anode","crime","salve","scout","imbue","pinky","stave","vague","chock","fight","video","stone","teach","cleft","frost","prawn","booty","twist","apnea","stiff","plaza","ledge","tweak","board","grant","medic","bacon","cable","brawl","slunk","raspy","forum","drone","women","mucus","boast","toddy","coven","tumor","truer","wrath","stall","steam","axial","purer","daily","trail","niche","mealy","juice","nylon","plump","merry","flail","papal","wheat","berry","cower","erect","brute","leggy","snipe","sinew","skier","penny","jumpy","rally","umbra","scary","modem","gross","avian","greed","satin","tonic","parka","sniff","livid","stark","trump","giddy","reuse","taboo","avoid","quote","devil","liken","gloss","gayer","beret","noise","gland","dealt","sling","rumor","opera","thigh","tonga","flare","wound","white","bulky","etude","horse","circa","paddy","inbox","fizzy","grain","exert","surge","gleam","belle","salvo","crush","fruit","sappy","taker","tract","ovine","spiky","frank","reedy","filth","spasm","heave","mambo","right","clank","trust","lumen","borne","spook","sauce","amber","lathe","carat","corer","dirty","slyly","affix","alloy","taint","sheep","kinky","wooly","mauve","flung","yacht","fried","quail","brunt","grimy","curvy","cagey","rinse","deuce","state","grasp","milky","bison","graft","sandy","baste","flask","hedge","girly","swash","boney","coupe","endow","abhor","welch","blade","tight","geese","miser","mirth","cloud","cabal","leech","close","tenth","pecan","droit","grail","clone","guise","ralph","tango","biddy","smith","mower","payee","serif","drape","fifth","spank","glaze","allot","truck","kayak","virus","testy","tepee","fully","zonal","metro","curry","grand","banjo","axion","bezel","occur","chain","nasal","gooey","filer","brace","allay","pubic","raven","plead","gnash","flaky","munch","dully","eking","thing","slink","hurry","theft","shorn","pygmy","ranch","wring","lemon","shore","mamma","froze","newer","style","moose","antic","drown","vegan","chess","guppy","union","lever","lorry","image","cabby","druid","exact","truth","dopey","spear","cried","chime","crony","stunk","timid","batch","gauge","rotor","crack","curve","latte","witch","bunch","repel","anvil","soapy","meter","broth","madly","dried","scene","known","magma","roost","woman","thong","punch","pasty","downy","knead","whirl","rapid","clang","anger","drive","goofy","email","music","stuff","bleep","rider","mecca","folio","setup","verso","quash","fauna","gummy","happy","newly","fussy","relic","guava","ratty","fudge","femur","chirp","forte","alibi","whine","petty","golly","plait","fleck","felon","gourd","brown","thrum","ficus","stash","decry","wiser","junta","visor","daunt","scree","impel","await","press","whose","turbo","stoop","speak","mangy","eying","inlet","crone","pulse","mossy","staid","hence","pinch","teddy","sully","snore","ripen","snowy","attic","going","leach","mouth","hound","clump","tonal","bigot","peril","piece","blame","haute","spied","undid","intro","basal","rodeo","guard","steer","loamy","scamp","scram","manly","hello","vaunt","organ","feral","knock","extra","condo","adapt","willy","polka","rayon","skirt","faith","torso","match","mercy","tepid","sleek","riser","twixt","peace","flush","catty","login","eject","roger","rival","untie","refit","aorta","adult","judge","rower","artsy","rural","shave","bobby","eclat","fella","gaily","harry","hasty","hydro","liege","octal","ombre","payer","sooth","unset","unlit","vomit","fanny","fetus","butch","stalk","flack","widow","augur"]

# ╔═╡ fa02baf4-dc36-461a-a7ea-0ee22fb6011f
@htl("""
Example Guess: $(@bind example_guess Select(nyt_valid_words; default = "apple"))
Example Answer: $(@bind example_answer Select(nyt_valid_words; default = "cloud"))
""")

# ╔═╡ 12d7468b-2675-476a-88ce-284e85b4a589
#index mapping valid guesses to a numerical index
const word_index = Dict(zip(nyt_valid_words, eachindex(nyt_valid_words)))

# ╔═╡ d3d33c45-434b-44ff-b4d7-7ba6e1f415fd
guess_random_word() = rand(nyt_valid_words)

# ╔═╡ a3121433-4d2e-4224-905b-aa0f8d91db05
md"""
### Precomputed Feedback Values and Distributions
"""

# ╔═╡ 82259951-eae3-4f20-89ea-b84144115028
md"""
### Visualization of the upper left 100 elements of the feedback matrix
"""

# ╔═╡ 8bb972da-20e5-4d0c-b964-b56ba62e631e
#bit vector representing all of the indices for valid guesses
const nyt_valid_inds = BitVector(fill(1, length(nyt_valid_words)))

# ╔═╡ 2b7adbb7-5c64-42fe-8178-3d1187f4b3fb
#bit vector representing all of the answers in the original wordle game
const wordle_original_inds = BitVector(in(nyt_valid_words[i], wordle_original_answers) for i in eachindex(nyt_valid_words))

# ╔═╡ e5b1d8e5-f224-44e3-8190-b8146ed3ea92
md"""
### Game Scoring Methodology Based on Information Gain

One way to think about the game is in terms of how many bits of information we gain for each guess.  In this context, information gain is the decrease in entropy of the distribution of what we believe are the remaining answers.  If we have $p_i = \Pr\{\text{guess}_i = \text{answer}\}$ for every guess, then that probability distribution can be used to compute the current entropy: $\text{entropy} = - \sum_i p_i \log p_i$.  For simplicity let's take our prior assumption of the answer distribution to just be a uniform distribution over the original answer words.  For the original game, the number of possible answers was $(length(wordle_original_answers)).  In the case of the uniform distribution $$p_i = \frac{1}{n}$$ where n is the number of non-zero items in the distribution.  So $$\text{entropy}_n = \log{n}$$ and in the case of the wordle original answers, this value is $(log2(length(wordle_original_answers))) bits of information when using the base 2 log.

After each guess, the provided information updates the distribution of possible answers by removing some from the original list, so on each step $n$ will either stay the same (in the case of no new information) or shrink.  When $n=1$, we know the answer and have gained all possible information from the distribution.  The possible answer distribution when we know the answer has the lowest possible entropy of zero.  In this case, we have gained the maximum amount of information possible which is all of the bits from the original distribution.

In the above function definitions are methods to compute the possible answer distribution after a series of guess/feedback pairs.  At any point in the game, this distribution can be computed and used to determine how much information has been gained up to this point and how much remains.  If a Wordle game could extend to an unlimited number of guesses, then we could simply score the game by dividing the information gain by the number of guesses required to win.  Since every game is eventually a win, the amount of information gained is a constant so this is equivalent to scoring shorter games higher and always favoring a faster finish.  In practice, it may make the most sense to allow lost games to continue until a hypothetical win to score policies, but this has the problem of potentially lasting forever.  One variation of the game called *hard mode* requires that guesses are words that still could be possible answers.  Forcing a policy to chose these words would eliminate the scenario where a game lasts forever.  Also, on the sixth guess of a game, if the guess is not the answer then the game is automatically lost, so a hard mode guess should always be forced on the sixth guess regardless of the policy preference.  So one possible approach to scoring games would be to force *hard mode* guesses on the sixth guess and on until the game is won.  By using this approach, lost games which are closer to an answer will be scored higher on average than lost games with many possible answers remaining and the score values are always based on the number of turns played.

Given that we are trying to end the game quickly, how should we make guesses in this very large action space?  One approach is to make guesses with the highest expected information gain based on the current distribution of possible answers.  The expected information gain can be quickly computed for every guess and used to rank guesses from best to worst.  There is one wrinkle with this scoring system though due to the distinction between a win and a loss.  An incorrect guess could narrow down the possible answer pool to 1 and thus provide the maximum possible information gain.  Another guess could do the same but also be the answer.  In the former case, an additional guess would be required in order to win while in the latter case we have won in one fewer turn.  One way to distinguish the cases would be to consider the information gain per guess.  For the cases where we guess the answer, we know just one additional guess was required to win the game.  In the case of an incorrect guess, we are left with a game with one or more possible answers.  If the number of possible answers is just 1, then the information gain will be the same, but the required guesses will be 2.  For all other cases, the number of guesses needed to win will be at least 1 more.  The information gain is smaller already as a starting point, but we do not know for certain how many guesses will be required to win.  If we assume on average just two more are needed then we would divide by 3.  For every scenario there will be a distribution of possible outcomes, so the decision on how badly to punish guesses that don't reveal all of the information will affect the ranking.  
"""

# ╔═╡ 32c30d8e-de5b-43a0-bfdb-8fb86037de7f
const wordle_original_entropy = log2(length(wordle_original_answers))

# ╔═╡ be061e94-d403-453a-99fb-7a1e13bebf52
md"""
To calculate the possible remaining answers, we must use the feedback information received from each guess.  A game state should contain this information in the form of a list of guesses and the corresponding feedback.
"""

# ╔═╡ 783d068c-77b0-43e1-907b-e532317c5afd
import Base.:(==)

# ╔═╡ bb5ef5ff-93fe-4985-a920-442862e4498b
import Base.hash

# ╔═╡ b9a5d547-79b4-4c7c-a3eb-ed7e87513a88
md"""
### Wordle Game State Definitions

To define a Wordle state I only need to know the list of guesses and corresponding feedback seen so far.  To save space and ensure that the two match I will use static vectors of the same length and only 8 bits per element for the feedback since the maximum value is 242.  For the guess list, I can use 16 bit unsigned integers because there are only about 15000 possible words.
"""

# ╔═╡ 1d5ba870-3110-4576-a116-a8d0a4d84edc
begin
	const wordle_actions = UInt16.(collect(eachindex(nyt_valid_words)))

	struct WordleState{N}
		guess_list::SVector{N, UInt16}
		feedback_list::SVector{N, UInt8}
	end

	#initialize a game start
	WordleState() = WordleState(SVector{0, UInt16}(), SVector{0, UInt8}())

	function WordleState(guess_list::AbstractVector{N}, feedback_list) where N<:Integer
		l = length(guess_list)
		WordleState(SVector{l}(UInt16.(guess_list)), feedback_list)
	end

	function WordleState(guess_list::AbstractVector{S}, feedback_list) where S<:AbstractString
		l = length(guess_list)
		WordleState(SVector{l}([UInt16(word_index[g]) for g in guess_list]), feedback_list)
	end

	function WordleState(guess_list::SVector{N, UInt16}, feedback_list::AbstractVector{T}) where {N, T<:Integer}
		l = length(feedback_list)
		WordleState(guess_list, SVector{l}(UInt8.(feedback_list)))
	end
	
	function Base.:(==)(s1::WordleState{N}, s2::WordleState{N}) where N
		(s1.guess_list == s2.guess_list) && (s1.feedback_list == s2.feedback_list)
	end

	Base.:(==)(s1::WordleState{N}, s2::WordleState{M}) where {N, M} = false

	const wordle_init_states = [WordleState()]

	get_possible_indices!(inds::BitVector, s::WordleState; kwargs...) = get_possible_indices!(inds, s.guess_list, s.feedback_list; kwargs...)

	get_possible_indices(s::WordleState; inds = copy(nyt_valid_inds), kwargs...) = get_possible_indices!(inds, s; kwargs...)

	#games are over after 6 guesses
	isterm(s::WordleState{6}) = true
	isterm(s::WordleState{0}) = false
	isterm(s::WordleState{N}) where N = (last(s.feedback_list) == 0xf2)
end

# ╔═╡ 93b857e5-72db-4aaf-abb0-295beab4073c
Base.hash(s::WordleState) = hash(s.guess_list) + hash(s.feedback_list) 

# ╔═╡ d2c1294e-1885-489c-98de-29930e9851c0
md"""
I will also define a Wordle afterstate which is just a Wordle state plus a guess.  This represents the game prior to receiving feedback for that guess.
"""

# ╔═╡ d7378cfd-140f-4161-a13f-398ffbc642b7
begin
struct WordleAfterState{N}
	state::WordleState{N}
	guess_index::UInt16
end
WordleAfterState(state::WordleState, guess_index::Integer) = WordleAfterState(state, UInt16(guess_index))
WordleAfterState(state::WordleState, guess::AbstractString) = WordleAfterState(state, UInt16(word_index[guess]))
end

# ╔═╡ 633e747b-3f5f-40e9-886a-590cb19d40dd
function Base.:(==)(w1::WordleAfterState, w2::WordleAfterState)
	(w1.state == w2.state) && (w1.guess_index == w2.guess_index)
end

# ╔═╡ 05b55c5f-0471-4032-9e7a-b61c58b33ce1
begin
	"""
		get_feedback(guess::SVector{5, Char}, answer::SVector{5, Char})
	"""
	function get_feedback(guess::SVector{5, Char}, answer::SVector{5, Char})
		output = zeros(UInt8, 5)
		counts = zeros(UInt8, 26)
	
		#green pass
		for (i, c) in enumerate(answer)
			j = letterlookup[c]
			#add to letter count in answer
			counts[j] += 0x01
			#exact match
			if c == guess[i]
				output[i] = EXACT
				#exclude one count from yellow pass
				counts[j] -= 0x01
			end
		end
	
		#yellow pass
		for (i, c) in enumerate(guess)
			j = letterlookup[c]
			if (output[i] == 0) && (counts[j] > 0)
				output[i] = MISPLACED
				counts[j] -= 0x01
			end
		end
		return SVector{5}(output)
	end

	"""
		get_feedback(guess::AbstractString, answer::AbstractString) -> SVector{5, UInt8}
	
	Compute feedback for a guess against an answer.
	
	Arguments
	
	- guess: A 5-character string guess
	- answer: A 5-character string answer
	
	Returns
	
	- A 5-element vector of feedback, where each element is one of:
	    - EXACT (exact match)
	    - MISPLACED (correct letter, wrong position)
	    - 0 (incorrect letter)
	"""
	get_feedback(guess::AbstractString, answer::AbstractString) = get_feedback(SVector{5, Char}(collect(lowercase(guess))), SVector{5, Char}(collect(lowercase(answer))))
end

# ╔═╡ b657a428-702a-47a3-9caa-7ed4ac09e7ca
const example_guess_feedback = get_feedback(example_guess, example_answer)

# ╔═╡ c43af2b6-733b-4b1b-a249-bb261f059783
const example_guess_feedback_bytes = convert_bytes(example_guess_feedback)

# ╔═╡ 6047122e-99bd-4719-b9fe-0253fe610780
#computes feedback in the form of a UInt16 number by encoding the terminary vector of length 5 into the corresponding number
function get_feedback_bytes(guess::SVector{5, UInt8}, answer::SVector{5, UInt8}; counts = zeros(UInt8, 26), output = zeros(UInt8, 5))
	feedback = zero(UInt16)
	counts .= zero(UInt8)
	output .= zero(UInt8)
	
	#green pass
	for (i, c) in enumerate(answer)
		#add to letter count in answer
		counts[c] += 0x01
		#exact match
		if c == guess[i]
			feedback += 0x002 * 0x003^(i-1)
			output[i] = EXACT
			#exclude one count from yellow pass
			counts[c] -= 0x01
		end
	end

	#yellow pass
	for (i, c) in enumerate(guess)
		if (output[i] == 0) && (counts[c] > 0)
			output[i] = MISPLACED
			feedback += 0x003^(i-1)
			counts[c] -= 0x01
		end
	end
	return UInt8(feedback)
end

# ╔═╡ e88225c3-931f-4672-8a38-ab58116a2b75
function make_feedback_sets(feedback_matrix)
	l = size(feedback_matrix, 1)
	out = Matrix{BitVector}(undef, l, 243)
	
	for feedback in 0:242
		for guess_index in 1:l
			out[guess_index, feedback+1] = BitVector(feedback_matrix[:, guess_index] .== feedback)
		end
	end
	#bit vectors that represent the indices of possible answers consistent with a combination of guess and feedback
	return out
end

# ╔═╡ d4f180c3-97f7-4e97-a904-a51f29207070
Base.hash(w::WordleAfterState) = hash(w.state) + hash(w.guess_index)

# ╔═╡ 4541253d-09fa-4485-80e0-0d64207be03d
begin
	"""
		make_feedback_matrix(word_list::Vector{T}) where T <: AbstractString -> Matrix{UInt16}
	
	Create a feedback matrix for a list of words.
	
	# Arguments
	
	- word_list: A vector of strings (words)
	
	# Returns
	
	- A matrix of feedback values, where each element feedback_matrix[i, j] represents the feedback for word i as a guess for word j
	
	# Details
	
	This function:
	
	- Converts each word to a vector of byte representations using letterlookup
	- Preallocates vectors for fast computation of feedback bytes
	- Computes the feedback matrix using get_feedback_bytes for each pair of words
	
	# See Also
	
	- `get_feedback_bytes`
	- `letterlookup`
	"""
	function make_feedback_matrix(word_list::AbstractVector{T}) where T<:AbstractString
		wordvecs = [make_word_vec(word) for word in word_list]
		make_feedback_matrix(wordvecs)
	end
	
	"""
		make_feedback_matrix(wordvecs::Vector{T}) where T<:SVector{5, UInt8}
	"""
	function make_feedback_matrix(wordvecs::Vector{T}) where T<:SVector{5, UInt8}
		l = lastindex(wordvecs)
		#preallocate counts and output for fast computation of feedback bytes
		counts = MVector{26, UInt8}(zeros(UInt8, 26))
		output = MVector{5, UInt8}(zeros(UInt8, 5))
		feedback_matrix = zeros(UInt8, l, l)
		for i in 1:l for j in 1:l
			feedback_matrix[i, j] = get_feedback_bytes(wordvecs[j], wordvecs[i]; counts = counts, output=output)
		end end
		return feedback_matrix
	end

	"""
		make_feedback_matrix(words::AbstractVector, savedata::Bool) -> Matrix{UInt16}

	Create a feedback matrix for a list of words, with the option of saving/loading a saved matrix
	
	"""
	function make_feedback_matrix(words::AbstractVector, savedata::Bool)
		fname = "feedback_matrix_$(hash(words)).bin"
		isfile(fname) && return read_feedback(fname, length(words))
		feedback_matrix = make_feedback_matrix(words)
		savedata && write(fname, feedback_matrix)
		return feedback_matrix
	end
end

# ╔═╡ 6ad205c6-7a67-4aa2-82a7-83358589ddca
#compute the feedback matrix and save the results, each column contains the feedback value for all of the possible answers for a given guess
const feedback_matrix = make_feedback_matrix(nyt_valid_words, false)

# ╔═╡ 46c0ce87-a94f-4b15-900f-be92775ee066
function make_feedback_sets(feedback_matrix, savedata::Bool)
	fname = "guess_feedback_lookup $(hash(feedback_matrix)).bin"
	l = size(feedback_matrix, 1)
	feedback_sets = if isfile(fname)
		open(fname) do f
			out = Matrix{BitVector}(undef, l, 243)
			feedbackset = BitVector(fill(false, l))
			for feedback in 1:243
				for guess_index in 1:l
					read!(f, feedbackset)
					out[guess_index, feedback] = copy(feedbackset)
				end
			end
			close(f)
			return out
		end
	else
		make_feedback_sets(feedback_matrix)
	end
	if savedata
		open(fname, "w") do f
			for feedback in 1:243
				for guess_index in 1:l
					write(f, feedback_sets[guess_index, feedback])
				end
			end
			close(f)
		end
	end
	return feedback_sets
end

# ╔═╡ f8d41dde-4b0d-432b-8223-d55bcce94736
const feedback_sets = make_feedback_sets(feedback_matrix, false)

# ╔═╡ 60c47b36-207d-4812-bcb8-4ecb932878ab
begin
	#for a given guess and feedback value, identify the indices of possible answers
	get_possible_indices(guess_index::Integer, feedback::Integer) = feedback_sets[guess_index, feedback+1]

	get_possible_indices(guess::AbstractString, feedback::Integer) = get_possible_indices(word_index[guess], feedback)

	get_possible_indices(guess, feedback::AbstractVector) = get_possible_indices(guess, convert_bytes(feedback))
end

# ╔═╡ a95fbf96-6996-4529-96cb-4761894f4c23
get_possible_words(args...) = nyt_valid_words[get_possible_indices(args...)]

# ╔═╡ e5e62c7e-d39b-47bf-b3c3-12b898816f51
#update an index BitVector with the possible answers consistent with a guess feedback pair
function get_possible_indices!(possible_answer_indices::BitVector, guess_index::Integer, feedback::Integer)
	possible_answer_indices .= get_possible_indices(guess_index, feedback)
	return possible_answer_indices
end

# ╔═╡ 4a8134aa-e0db-43ad-837d-c38a235b4712
function count_possible(guess_index, feedback, base_indices)
	new_possible_indices = get_possible_indices(guess_index, feedback)
	dot(new_possible_indices, base_indices)
end

# ╔═╡ c1858879-2a20-4af6-af4c-a03d26dda7a3
#figure out which guess we would expectc produce the largest increase in information given a state
function eval_guess_information_gain(possible_answer_inds::BitVector, guess_count::Integer; possible_answers = copy(wordle_actions), save_all_scores = false)
	l = sum(possible_answer_inds)
	iszero(l) && error("No possible answers left")

	# (l == 1) && error("Only the answer $(nyt_valid_words[findfirst(possible_answer_inds)]) remains.  No guesses to assess")

	(l == 1) && return (best_guess = wordle_actions[findfirst(possible_answer_inds)],)
	
	# possible_answers = wordle_actions[possible_answer_inds]
	view(possible_answers, 1:l) .= view(wordle_actions, possible_answer_inds)

	if save_all_scores
		guess_scores = zeros(Float32, length(nyt_valid_words))
		expected_entropy = zeros(Float32, length(nyt_valid_words))
	end

	best_guess = 1
	best_score = 0f0
	starting_entropy = Float32(wordle_original_entropy)
	best_entropy = Float32(starting_entropy)
	
	if l > 1000
		feedback_entropies = zeros(Float32, 243)
		for guess_index in wordle_actions
			final_entropy = 0.0f0
			score = 0.0f0
			# @fastmath @inbounds @simd for i in view(possible_answers, 1:l)
			@fastmath @inbounds for f in 0x00:0xf2
				n = dot(get_possible_indices(guess_index, f), possible_answer_inds)
				feedback_entropies[f+1] = Float32(log2(n))
			end
			
			@fastmath @inbounds @simd for i in view(possible_answers, 1:l)
				#for the case where the guess is the answer, this will reduce the entropy to zero, but I want to treat that differently than the case where the guess is incorrect and there is only one possible remaining answer.  the latter case requires one additional guess minimum while the former case ends the game immediately.  I could keep track of information gain per incorrect guess + 1 in which case all the other values would be divided by 2.  But this metric has to be valid for an expected value by averaging up the results.  This function will evaluate the final score for a word from the original state so that means accounting for the total entropy gain from the start.  Assuming the game start was the original wordle words, this starting entropy is just log2(length(wordle_original_answers)).  Then I will calculate the total entropy gain and divide by the minimum number of turns before the game ends.  This will be an underestimate of the true best score since a certain information gain guess that doesn't end the game will have a score as if no further information gain occurs and the game lasts an additional turn at least
				f = feedback_matrix[i, guess_index]
				# if feedback_entropies[f + 0x01] == -1
					# n = dot(get_possible_indices(guess_index, f), possible_answer_inds)
					# entropy = Float32(log2(n))
				# else
					# entropy = feedback_entropies[f + 0x01]
				# end
				entropy = feedback_entropies[f+1]
				information_gain = starting_entropy - entropy
				win = (guess_index == i)
				oneleft = iszero(entropy)
				d = guess_count + win + 2*(oneleft*!win) + 3*(!oneleft)
				score += information_gain / d
				final_entropy += entropy
			end
	
			final_entropy /= l
			score /= l
			if score > best_score
				best_score = score
				best_entropy = final_entropy
				best_guess = guess_index
			end
	
			if save_all_scores
				guess_scores[guess_index] = score
				expected_entropy[guess_index] = final_entropy
			end
		end
	else
		answer_entropies = zeros(Float32, l)
		for guess_index in wordle_actions
			final_entropy = 0.0f0
			score = 0.0f0
			@fastmath @inbounds for i in 1:l
				answer_index = possible_answers[i]
				f = feedback_matrix[answer_index, guess_index]
				n = dot(get_possible_indices(guess_index, f), possible_answer_inds)
				entropy = Float32(log2(n))
				answer_entropies[i] = entropy
			end
			@fastmath @inbounds @simd for i in 1:l
				#for the case where the guess is the answer, this will reduce the entropy to zero, but I want to treat that differently than the case where the guess is incorrect and there is only one possible remaining answer.  the latter case requires one additional guess minimum while the former case ends the game immediately.  I could keep track of information gain per incorrect guess + 1 in which case all the other values would be divided by 2.  But this metric has to be valid for an expected value by averaging up the results.  This function will evaluate the final score for a word from the original state so that means accounting for the total entropy gain from the start.  Assuming the game start was the original wordle words, this starting entropy is just log2(length(wordle_original_answers)).  Then I will calculate the total entropy gain and divide by the minimum number of turns before the game ends.  This will be an underestimate of the true best score since a certain information gain guess that doesn't end the game will have a score as if no further information gain occurs and the game lasts an additional turn at least
				# f = feedback_matrix[i, guess_index]
				# if feedback_entropies[f + 0x01] == -1
					# n = dot(get_possible_indices(guess_index, f), possible_answer_inds)
					# entropy = Float32(log2(n))
				# else
					# entropy = feedback_entropies[f + 0x01]
				# end
				entropy = answer_entropies[i]
				information_gain = starting_entropy - entropy
				answer_index = possible_answers[i]
				win = (guess_index == answer_index)
				oneleft = iszero(entropy)
				d = guess_count + win + 2*(oneleft*!win) + 3*(!oneleft)
				score += information_gain / d
				final_entropy += entropy
			end
	
			final_entropy /= l
			score /= l
			if score > best_score
				best_score = score
				best_entropy = final_entropy
				best_guess = guess_index
			end
	
			if save_all_scores
				guess_scores[guess_index] = score
				expected_entropy[guess_index] = final_entropy
			end
		end
	end

	best_guess_out = (best_guess = best_guess, best_score = best_score, best_guess_entropy = best_entropy)

	!save_all_scores && return best_guess_out
	
	(guess_scores = guess_scores, expected_entropy = expected_entropy, ranked_guess_inds = sortperm(guess_scores, rev=true))
end

# ╔═╡ b75bb4fe-6b09-4d7c-8bc0-6bf224d2c95a
function get_possible_indices!(inds::BitVector, guess_list::SVector{N, UInt16}, feedback_list::SVector{N, UInt8}; baseline = wordle_original_inds) where N
	inds .= baseline
	for i in 1:N
		inds .*= get_possible_indices(guess_list[i], feedback_list[i])
	end
	return inds
end

# ╔═╡ 852dff48-8c1d-42f2-925e-53f889b1ac6a
begin
	const greedy_information_gain_lookup = Dict{WordleState, @NamedTuple{best_guess::Int64, best_score::Float32, best_guess_entropy::Float32}}()
	const all_guesses_information_gain_lookup = Dict{WordleState, @NamedTuple{guess_scores::Vector{Float32}, expected_entropy::Vector{Float32}, ranked_guess_inds::Vector{Int64}}}()
end

# ╔═╡ 4eb18ec4-b327-4f97-a380-10469441cff8
function eval_guess_information_gain(s::WordleState{N}; possible_indices = copy(nyt_valid_inds), save_all_scores = false, kwargs...) where N
	save_all_scores && haskey(all_guesses_information_gain_lookup, s) && return all_guesses_information_gain_lookup[s]
	!save_all_scores && haskey(greedy_information_gain_lookup, s) && return greedy_information_gain_lookup[s]	
		
	get_possible_indices!(possible_indices, s)
	out = eval_guess_information_gain(possible_indices, N; save_all_scores = save_all_scores, kwargs...)

	(length(out) == 1) && return out
	
	if save_all_scores
		all_guesses_information_gain_lookup[s] = out
	else
		greedy_information_gain_lookup[s] = out
	end
	
	return out
end

# ╔═╡ 20a41989-6650-41dd-b34e-fa23c993e669
const state_entropy_lookup = Dict{WordleState, Float32}()

# ╔═╡ d9c4ff04-12ba-4e1c-bc52-4c12388d514b
function calculate_state_entropy(s::WordleState; possible_indices = copy(nyt_valid_inds))
	haskey(state_entropy_lookup, s) && return state_entropy_lookup[s]
	get_possible_indices!(possible_indices, s)
	n = sum(possible_indices)
	state_entropy_lookup[s] = Float32(log2(n))
end

# ╔═╡ 97d40991-d719-4e87-bd70-94551a645448
function display_one_step_guesses(output)
	[(word = nyt_valid_words[i], score = output.guess_scores[i], expected_entropy = output.expected_entropy[i]) for i in output.ranked_guess_inds]
end

# ╔═╡ 92cb4e96-714d-425d-a64b-eff26a5f92ef
md"""
Using this one step lookahead method to rank the guesses, we can already come up with a candidate policy to test, one that is greedy with respect to this score.
"""

# ╔═╡ a383cd80-dd49-4f9f-ba67-1f58ea7eb0b6
@skip_as_script wordle_start_information_gain_guesses = eval_guess_information_gain(WordleState(); save_all_scores = true) |> display_one_step_guesses

# ╔═╡ 2cda2a2c-caf2-474c-90a4-388f15260501
eval_guess_information_gain(WordleState(["trace"], [convert_bytes([0, 1, 1, 0, 0])]); save_all_scores = true) |> display_one_step_guesses

# ╔═╡ f2ab3716-19a2-4cbb-b46f-a13c297e086d
md"""
### Wordle Game Dynamics

In order to simulate games we must have a way of producing the appropriate feedback and taking a game to completion.  By using this type of afterstate MDP all of the transitions are actually deterministic since I have the full probability distribution for every possible outcome
"""

# ╔═╡ 86f6910f-0187-4d38-afff-295ec67fd0c0
#simple function just adds the guess to an existing wordle state
function wordle_afterstate_step(s::WordleState, a::UInt16)
	isterm(s) && error("Cannot make guess from terminal state $s")
	(0f0, WordleAfterState(s, a))
end

# ╔═╡ 35025fe1-e98b-46c6-b159-37d255100b73
const wordle_initial_afterstates = [wordle_afterstate_step(WordleState(), a) for a in wordle_actions]

# ╔═╡ b1e3f7ea-e2de-4467-b301-0c3cf225e433
const wordle_afterstate_transition_lookup = Dict{WordleAfterState, Dict{Tuple{Float32, WordleState}, Float32}}()

# ╔═╡ 8e2477a9-ca14-4ecf-9194-9bac5ef25fd4
md"""
### Visualize Afterstate Transition
"""

# ╔═╡ 961593ef-964c-4363-b4ba-0d45cfe3d198
const test_possible_indices = copy(nyt_valid_inds)

# ╔═╡ f6253866-7bef-400f-9713-e0fd5054b201
scoregame(s::WordleState{6}) = Float32((wordle_original_entropy - calculate_state_entropy(s)) / (7f0 - (last(s.feedback_list) == 0xf2)))

# ╔═╡ cda5622b-f4f7-4ec2-a848-ff5eb9c9b66b
function scoregame(s::WordleState{N}) where N
	!isterm(s) && return 0f0
	Float32(wordle_original_entropy / N)
end

# ╔═╡ bd744b6d-db57-42bf-88c0-c667dfb03f5f
#produces the distribution of possible feedback and thus transition states given an afterstate
function wordle_afterstate_transition(w::WordleAfterState{N}; possible_indices = copy(nyt_valid_inds), possible_answers = wordle_original_inds, use_cache = true) where N
	(use_cache && haskey(wordle_afterstate_transition_lookup, w)) && return wordle_afterstate_transition_lookup[w]
	get_possible_indices!(possible_indices, w.state; baseline = possible_answers)
	n = sum(possible_indices)
	output = zeros(Float32, 243)
	for answer_index in view(wordle_actions, possible_indices)
		f = feedback_matrix[answer_index, w.guess_index]
		output[f+1] += 1f0
	end
	output ./= n
	dist = Dict{Tuple{Float32, WordleState}, Float32}()
	new_guess_list = SVector{N+1}([w.state.guess_list; w.guess_index])
	for i in eachindex(output)
		if output[i] != 0
			s′ = WordleState(new_guess_list, SVector{N+1}([w.state.feedback_list; UInt8(i-1)]))
			r = scoregame(s′)
			dist[(r, s′)] = output[i]
		end
	end
	use_cache && (wordle_afterstate_transition_lookup[w] = dist)
	return dist
end

# ╔═╡ a3b9f0c9-c8bd-4237-8b1c-a1fed136e988
create_wordle_afterstate_mdp() = AfterstateMDP{Float32, WordleState, WordleAfterState}(() -> WordleState(), wordle_actions, wordle_afterstate_step, wordle_afterstate_transition, isterm)

# ╔═╡ d3d0df67-ec26-4352-9164-a53d14d1b065
const wordle_afterstate_mdp = create_wordle_afterstate_mdp()

# ╔═╡ 9a75b05d-82fd-4a3c-8cb9-5161dfc18949
function wordle_greedy_information_gain_π(s::WordleState; kwargs...)
	score = eval_guess_information_gain(s; kwargs...)
	UInt16(score.best_guess)
end

# ╔═╡ 73d4533a-1d8a-4b3a-a4ed-a9f64cc5ee23
function wordle_afterstate_rollout(s::WordleState, π::Function; possible_indices = copy(nyt_valid_inds))
	isterm(s) && return 0f0
	a = π(s) #get action from policy
	(r, w) = wordle_afterstate_step(s, a) #produce single afterstate
	dist = wordle_afterstate_transition(w; possible_indices = possible_indices) #get distribution of states from afterstate
	# r′ = keys(dist) |> Map(k -> dist[k]*(k[1] + wordle_afterstate_rollout(k[2], π; possible_indices = possible_indices))) |> foldxl(+)
	# r + r′
	# r′ = r
	# for k in keys(dist)
	# 	r′ += dist[k]*k[1]
	# 	future_reward = wordle_afterstate_rollout(k[2], π; possible_indices = possible_indices)
	# 	r′ += dist[k]*future_reward
	# end
	# return r′
	r + sum(dist[k]*(k[1] + wordle_afterstate_rollout(k[2], π; possible_indices = possible_indices)) for k in keys(dist)) #average the result of a rollout on all the transition states weighted by their relative probabilities
end

# ╔═╡ 05ffe4f9-8cd6-424e-8955-e3ca21c486f3
function wordle_greedy_information_gain_state_value(s::WordleState; possible_indices = copy(nyt_valid_inds), possible_answers = copy(wordle_actions))
	π(s) = wordle_greedy_information_gain_π(s; possible_indices = possible_indices, possible_answers = possible_answers)
	wordle_afterstate_rollout(s, π; possible_indices = possible_indices)
end

# ╔═╡ b0e2ea69-13a9-4520-920b-bf5089cc8da8
function wordle_greedy_information_gain_afterstate_value(w::WordleAfterState)
	possible_indices = copy(nyt_valid_inds)
	possible_answers = copy(wordle_actions)
	π(s) = wordle_greedy_information_gain_π(s; possible_indices = possible_indices, possible_answers = possible_answers)
	dist = wordle_afterstate_transition(w)
	# out = 0f0
	# for k in keys(dist)
	# 	r = wordle_afterstate_rollout(k[2], π; possible_indices = possible_indices)
	# 	out += dist[k]*(k[1] + r)
	# end
	# return out
	sum(dist[k]*(k[1] + wordle_afterstate_rollout(k[2], π)) for k in keys(dist))
end

# ╔═╡ 28709a4e-67cd-4dff-aad3-cb8af93038d7
begin
	wordle_greedy_information_gain_guess_value(s::WordleState, guess_index::Integer) = wordle_greedy_information_gain_afterstate_value(WordleAfterState(s, guess_index))
	wordle_greedy_information_gain_guess_value(s::WordleState, guess::AbstractString) = wordle_greedy_information_gain_guess_value(s, word_index[guess])
end

# ╔═╡ 379e4103-ca2a-4711-9df7-d6cd83afe653
const score_values = [(winning_turn = n, score = wordle_original_entropy / n) for n in 1:6]

# ╔═╡ 8cb865b4-8928-4210-96d5-6d6a71deb03b
#evaluation of starting state using greedy information gain policy
wordle_greedy_information_gain_state_value(WordleState())

# ╔═╡ 09c89ae6-83e2-4f86-8c3f-b1528235c70a
#expected score for random game
wordle_afterstate_rollout(WordleState(), s -> rand(wordle_actions))

# ╔═╡ 2fa84583-4ff7-48d0-bd62-cb7380535e91
function wordle_greedy_information_gain_one_step_policy_iteration(s; num_evaluations = 10)
	best_guess = 1
	best_rank = 1
	best_score = 0f0
	t0 = time()
	last_time = time()
	output = zeros(Float32, num_evaluations)

	guess_scores = eval_guess_information_gain(s; save_all_scores = true)
	scoreinds = view(guess_scores.ranked_guess_inds, 1:num_evaluations)
	hard_mode_indices = get_possible_indices(s; baseline = nyt_valid_inds)
	likely_answer_indices = get_possible_indices(s)
	num_hard_mode = sum(hard_mode_indices)
	num_likely = sum(likely_answer_indices)
	for (i, guess_index) in enumerate(scoreinds)
		v = wordle_greedy_information_gain_guess_value(s, guess_index)
		output[i] = v
		elapsed = time() - t0
		pct_done = i / length(scoreinds)
		estimated_total_time = elapsed / pct_done
		if v > best_score
			best_guess = guess_index
			best_score = v
			best_rank = i
		end
		if (time() - last_time) > 5
			last_time = time()
			@info """Guess number $i of $(length(scoreinds)) is $(nyt_valid_words[guess_index]) with a score of $v
			After $(round(Int64, elapsed / 60)) minutes best guess is $(nyt_valid_words[best_guess]) with a rank of $best_rank and a score of $best_score. ETA=$(round(Int64, estimated_total_time/60)) minutes"""
		end
	end
	sorted_inds = sortperm(output; rev=true)
	(best_guess = best_guess, best_value = best_score, hard_mode_indices = hard_mode_indices, likely_answer_indices = likely_answer_indices, sorted_guesses = [(guess = nyt_valid_words[guess_index], guess_index = guess_index, expected_value = output[i], information_gain_rank = i, information_gain_score = guess_scores.guess_scores[guess_index], expected_entropy = guess_scores.expected_entropy[guess_index], answer_probability = likely_answer_indices[guess_index] / num_likely, hard_mode_guess = Bool(hard_mode_indices[guess_index])) for (i, guess_index) in enumerate(scoreinds)][sorted_inds])
end

# ╔═╡ 4578c627-127e-4a18-972c-03b38cff371b
md"""
#### Guess Values for Greedy Information Policy from Starting State
"""

# ╔═╡ 52e5d54d-c40d-48e1-943f-01985e9c4f7f
const new_test_state = WordleState(["trace"], [convert_bytes([0, 1, 1, 0, 0])])

# ╔═╡ e804f5fc-c817-4851-a14b-fdcca1180773
function evaluate_wordle_state(s::WordleState; num_evaluations = 100, filter_hard = false)
	output = wordle_greedy_information_gain_one_step_policy_iteration(s; num_evaluations = num_evaluations)
	score_table = output.sorted_guesses |> DataFrame
	hard_mode_words = nyt_valid_words[output.hard_mode_indices]
	answer_words = nyt_valid_words[output.likely_answer_indices]

	if filter_hard
		filter!(a -> a.hard_mode_guess, score_table)
	end
	
	hard_mode_display = if length(hard_mode_words) < 20	
		wordlist = mapreduce(w -> in(w, answer_words) ? uppercase(w) : w, (w1, w2) -> "$w1 $w2", hard_mode_words)
		md"""
		Likely answers shown in caps: $wordlist
		"""
	else
		md""""""
	end
	
	md"""
	##### Words Remaining: 
	
	Hard Mode = $(length(hard_mode_words)), Likely Answers = $(length(answer_words))

	$hard_mode_display
	
	##### Ranked Guesses:

	$score_table
	"""
end

# ╔═╡ 7f1f2057-24ca-44c0-8496-b6ef68d895bd
md"""
### Defining Afterstate MCTS
"""

# ╔═╡ 9ba4f6e0-f0cc-4b8f-b70e-8e9a06aa03ad
function wordle_update_gumbel_information_tree_policy!(v, s::WordleState{N}; topn = 10, score_vector = zeros(Float32, length(nyt_valid_inds)), possible_indices = copy(nyt_valid_inds), possible_answers = copy(wordle_actions), p_scale::Float32 = 100f0) where N
#here v is a vector with indices representing the action space and s is the current state. by default v is filled with the existing q value estimates plus the UCT bonus and we would select the action greedily.  To match the behavior of the rollout though, here we make a few changes to the action selection.  If there have already been 5 guesses, then the tree policy will always choose from one of the remaining possible answers.
	get_possible_indices!(possible_indices, s)
	n = sum(possible_indices)
	l = N
	if (l == 5) || (n <= 2) #for the final guess and if only two answers remain only consider possible answers
		v .*= possible_indices
		make_greedy_policy!(v)
	else
		sampled_inds = findall(!isinf, v) #these are actions that have already been sampled, always include them
		# if (length(sampled_inds) < min_samples_explore) || (rand() < explore_probability) #some percent of the time, sample new indices
			information_scores = eval_guess_information_gain(s; possible_indices = possible_indices, possible_answers = possible_answers, save_all_scores = true)

			min_score = information_scores.guess_scores[last(information_scores.ranked_guess_inds)]
			guess_scores = information_scores.guess_scores
			
			#convert scores to log probabilities and add gumbel random variable, here c is used to amplify the effect of the score to weight the probability more towards higher scores
			score_vector .= p_scale .* log.(guess_scores .- min_score) .- log.(-log.(rand(Float32, length(v))))
			# @inbounds @fastmath @simd for i in eachindex(one_step_scores)
				# score_vector[i] = 100 * log(one_step_scores[i]) - log(-log(rand(Float32)))
			# end
			
			include_indices = [partialsortperm(score_vector, 1:topn; rev=true); sampled_inds]
			v_select = argmax(v[i] for i in include_indices)
			# score_vector .= 0f0
			# score_vector[include_indices] .= v[include_indices]
			# v .= score_vector
			v .= 0f0
			v[include_indices[v_select]] = 1f0
			# exclude_indices = partialsortperm(score_vector, 1:length(v)-topn)
			# v_include = v[include_indices]
			# one_step_sort_inds = partialsortperm(score_vector; rev=true)
			# exclude_indices = view(one_step_sort_inds, 1:topn) #only sample the top n values which is equivalent to sampling n values without replacement from the distribution of scores
			# v[exclude_indices] .= 0f0 #set these to a value lower than any estimate with samples
			# v .= 0f0
			# v[include_indices] .= v_include
		# else
			# unsampled_inds = findall(isinf, v) #these are all actions that have not yet been sampled
			# v[unsampled_inds] .= 0f0 #99% of the time only consider actions that have already been sampled
			# make_greedy_policy!(v)
		# end
	end
	# make_greedy_policy!(v)
	return v
end

# ╔═╡ e093dd57-040b-4983-a95f-057097fcefff
function run_wordle_afterstate_mcts(s::WordleState, nsims::Integer; topn = 10, p_scale = 100f0, kwargs...)
	possible_indices = copy(nyt_valid_inds)
	possible_answers = copy(wordle_actions)
	score_vector = zeros(Float32, length(nyt_valid_inds))
	(mcts_guess, tree_values) = monte_carlo_tree_search(wordle_afterstate_mdp, 1f0, (mdp, s, γ) -> wordle_greedy_information_gain_state_value(s; possible_indices = possible_indices, possible_answers = possible_answers), s; nsims = nsims, depth = 10_000, c = 1f0, update_tree_policy! = (v, s) -> wordle_update_gumbel_information_tree_policy!(v, s; possible_indices = possible_indices, possible_answers = possible_answers, score_vector = score_vector, topn = topn, p_scale = p_scale), sim_message = true, kwargs...)
	return tree_values
end

# ╔═╡ 06138a4e-67a3-47a3-84a5-92013fd404ca
const tree_values = run_wordle_afterstate_mcts(WordleState(), 2)

# ╔═╡ 10d1f403-34c8-46ce-8cfc-d289608f465c
const track_run, set_run = @use_state("Completed")

# ╔═╡ 03bbb910-dfa7-4c28-b811-afa9e5ca0e63
track_run

# ╔═╡ 6fdc99b1-beea-4c45-98bb-d257501a6878
function show_afterstate_mcts_guesses(tree_values, s::WordleState)
	new_state_dict = tree_values[s][2]
	vest = tree_values[s][1]
	(vest = vest, ranked_guesses = sort([(word = nyt_valid_words[k], visits = new_state_dict[k][1], values = new_state_dict[k][2]/new_state_dict[k][1]) for k in keys(new_state_dict)]; by = t -> t.values, rev=true))
end

# ╔═╡ 889a2d4b-1adf-44cd-8d3f-22d4aa4c3de3
const base_mcts_options = show_afterstate_mcts_guesses(tree_values, WordleState())

# ╔═╡ 87e696f6-a9f5-4e00-8678-ec81ad530dd0
const mcts_options, set_mcts_options = @use_state(base_mcts_options)

# ╔═╡ cb3b46cf-8375-43b2-8736-97882e9b5e18
mcts_options

# ╔═╡ aaf516b5-f982-44c3-bcae-14d46ad72e82
md"""
# Dependencies
"""

# ╔═╡ ee646ead-3600-49ed-b7ed-3c9a8af7d195
md"""
# Wordle Visualization
"""

# ╔═╡ 58316d89-7149-42fe-802f-ecd44999ea77
md"""
## HTML Elements
"""

# ╔═╡ 731c6a58-b223-48f3-82dc-b7cf3772c699
function stylelabel(letter)
	"""
	.inputbox[label=$letter]::before {
		content: '$letter';
	}
	.inputbox[label=$letter] {
		border-color: rgb(86, 87, 88);
	}
	"""
end

# ╔═╡ 028ad58f-f34b-4741-a99e-42c86f88bd74
const nyt_keyboard = """
	<div class="Keyboard-module_keyboard__uYuqf" role="group" aria-label="Keyboard"><div class="Keyboard-module_row__ilOKU"><button type="button" data-key="q" class="Key-module_key__kchQI">q</button><button type="button" data-key="w" class="Key-module_key__kchQI">w</button><button type="button" data-key="e" class="Key-module_key__kchQI">e</button><button type="button" data-key="r" class="Key-module_key__kchQI">r</button><button type="button" data-key="t" class="Key-module_key__kchQI">t</button><button type="button" data-key="y" class="Key-module_key__kchQI">y</button><button type="button" data-key="u" class="Key-module_key__kchQI">u</button><button type="button" data-key="i" class="Key-module_key__kchQI">i</button><button type="button" data-key="o" class="Key-module_key__kchQI">o</button><button type="button" data-key="p" class="Key-module_key__kchQI">p</button></div><div class="Keyboard-module_row__ilOKU"><div data-testid="spacer" class="Key-module_half__HooWu"></div><button type="button" data-key="a" class="Key-module_key__kchQI">a</button><button type="button" data-key="s" class="Key-module_key__kchQI">s</button><button type="button" data-key="d" class="Key-module_key__kchQI">d</button><button type="button" data-key="f" class="Key-module_key__kchQI">f</button><button type="button" data-key="g" class="Key-module_key__kchQI">g</button><button type="button" data-key="h" class="Key-module_key__kchQI">h</button><button type="button" data-key="j" class="Key-module_key__kchQI">j</button><button type="button" data-key="k" class="Key-module_key__kchQI">k</button><button type="button" data-key="l" class="Key-module_key__kchQI">l</button><div data-testid="spacer" class="Key-module_half__HooWu"></div></div><div class="Keyboard-module_row__ilOKU"><button type="button" data-key="↵" class="Key-module_key__kchQI Key-module_oneAndAHalf__bq8Tw">enter</button><button type="button" data-key="z" class="Key-module_key__kchQI">z</button><button type="button" data-key="x" class="Key-module_key__kchQI">x</button><button type="button" data-key="c" class="Key-module_key__kchQI">c</button><button type="button" data-key="v" class="Key-module_key__kchQI">v</button><button type="button" data-key="b" class="Key-module_key__kchQI">b</button><button type="button" data-key="n" class="Key-module_key__kchQI">n</button><button type="button" data-key="m" class="Key-module_key__kchQI">m</button><button type="button" data-key="←" aria-label="backspace" class="Key-module_key__kchQI Key-module_oneAndAHalf__bq8Tw"><svg xmlns="http://www.w3.org/2000/svg" height="20" viewBox="0 0 24 24" width="20" class="game-icon" data-testid="icon-backspace"><path fill="var(--color-tone-1)" d="M22 3H7c-.69 0-1.23.35-1.59.88L0 12l5.41 8.11c.36.53.9.89 1.59.89h15c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm0 16H7.07L2.4 12l4.66-7H22v14zm-11.59-2L14 13.41 17.59 17 19 15.59 15.41 12 19 8.41 17.59 7 14 10.59 10.41 7 9 8.41 12.59 12 9 15.59z"></path></svg></button></div></div>
"""

# ╔═╡ 15bc3ba3-8fa7-4c62-91b6-71e797e4a83d
const wordlegamestyle = HTML("""
<style>
	.buttons {
		width: var(--container-width);
		margin: 5px;
		display: flex;
		justify-content: left;
	}
	.buttons * {
		font-size: 1em;
	}
	.wordle-game, .wordle-game-input {
		display: flex;
		flex-direction: column;
		align-items: center;
	}
	.wordle-game-grid {
		display: grid;
		width: var(--container-width);
		height: calc(var(--container-width)*6/5);
		grid-template-columns: repeat(5, 1fr);
		grid-template-rows: repeat(6, 1fr);
		box-sizing: border-box;
		margin: calc(var(--container-width)/35);
		padding: 0;
		grid-gap: calc(var(--container-width)/70);
		position: relative;
	}
	.letterGrid {
		display: grid;
		grid-template-columns: repeat(10, auto);
		column-gap: calc(var(--container-width)/130);
		row-gap: calc(var(--container-width)/130);
		justify-content: center;
	}
	.letter.u {
		grid-column-start: 3;
	}
	.letter {
		border: 2px solid black;
		aspect-ratio: 1/1;
		height: calc(var(--container-width)/13);
		display: flex;
		justify-content: center;
		align-items: center;
		background-color: rgb(110, 110, 110);
		border-radius: 20%;
		font-family: "Arial", sans-serif;
		font-weight: bold;
		-webkit-font-smoothing: antialiased;
		text-transform: uppercase;
		font-size: calc(var(--container-width)/5/6.0); 
	}

	.Keyboard-module_keyboard__uYuqf {
	  height: calc(var(--container-width)*2/3.5);
	  width: min(90vw, calc(var(--container-width)*1.3));
	  margin: 0 8px;
	  -webkit-user-select: none;
	  -moz-user-select: none;
	  user-select: none;
	  padding: 0;
	  border: 0;
	}
	
	.Keyboard-module_row__ilOKU {
	  display: flex;
	  width: 100%;
	  margin: 0 auto 8px;
	  touch-action: manipulation;
	  padding: 0;
	  border: 0;
	}
	
	.Key-module_half__HooWu {
	  flex: .5;
	}
	
	.Key-module_key__kchQI {
	  font-family: "Arial";
	  font-size: 1.25em;
	  font-weight: bold;
	  border: 0;
	  padding: 0;
	  margin: 0 6px 0 0;
	  height: calc(var(--container-width)*1.3*.12);
	  border-radius: 4px;
	  cursor: pointer;
	  -webkit-user-select: none;
	  -moz-user-select: none;
	  user-select: none;
	  background-color: #818384;
	  color: #ffffff;
	  flex: 1;
	  display: flex;
	  justify-content: center;
	  align-items: center;
	  text-transform: uppercase;
	  -webkit-tap-highlight-color: rgba(0,0,0,.3);
	}
	
	.Key-module_oneAndAHalf__bq8Tw {
	  flex: 1.5;
	  font-size: 12px;
	}
</style>
""")

# ╔═╡ 9c898c04-65a3-44b0-bbfa-bfcf4ca32c0f
function add_style(block::String)
	HTML("""
	<style>
	$block
	</style>
	""")
end

# ╔═╡ a78a462c-647a-477d-abcc-aaeae1202cca
add_style(
	"""
	.Key-module_key__kchQI:active, .Key-module_key__kchQI.pressed {
		transform: translateY(3px) scale(.98);
		filter: brightness(0.85);
	}

	.Key-module_key__kchQI:hover:not(.Key-module_key__kchQI:active) {
		animation: hover-key .85s infinite;
	}

	@keyframes hover-key {
		50% {filter: brightness(1.2); transform: translateY(-1px) scale(1.01);}
	}
	
	"""
)

# ╔═╡ 9cf9cb1c-62af-4dde-b28b-d6733c1432b0
const colorlookup = Dict([0x00 => "#3a3a3c", 0x01 => "#b59f3b", 0x02 => "#538d4e"])

# ╔═╡ 7c968494-0e3e-47e4-9666-6d1f7e5a9e85
const final_box_style = """color: #ffffff;"""

# ╔═╡ 37209bdb-8f5c-4e4b-8e88-c77a6a07fdfd
begin
function add_elements(a::AbstractString, b::AbstractString)
	"""
	$a
	$b
	"""
end
add_elements(a::HTML, b::HTML) = add_elements(a.content, b.content)
add_elements(a::HTML, b::AbstractString) = add_elements(a.content, b)
add_elements(a::AbstractString, b::HTML) = add_elements(a, b.content)
end

# ╔═╡ f79720db-a5dc-4267-a209-181e15e6f200
const basewordlestyle = HTML(
	"""
	<style>
		:root {
			--container-width: min(500px, 90vw);
		}
	
		.wordle-box {
			display: grid;
			grid-template-columns: repeat(5, 1fr);
			grid-gap: calc(var(--container-width)/60);
			width: var(--container-width);
			height: calc(var(--container-width)/5);
			margin: calc(var(--container-width)/15);
			position: relative;
		}
		
		.inputbox {
			display: inline-flex;
			width: 100%;
			aspect-ratio: 1/1;
			vertical-align: middle;
			justify-content: center;
			align-items: center;
			margin: 0;
			padding: 0;
			font-family: "Arial", sans-serif;
			font-weight: bold;
			font-stretch: 100%;
			-webkit-font-smoothing: antialiased;
			text-transform: uppercase;
			font-size: calc(var(--container-width)/5/2.0); 
			background-color: rgba(0, 0, 0, 0);
			$final_box_style;
			border: 2px solid #3a3a3c;
		}

		.inputbox.invalid {
			animation: shake 600ms;
		}

		@keyframes shake {
			10%,90% {transform: translateX(-1px);}
			20%,80% {transform: translateX(-2px);}
			30%,50%,70% {transform: translateX(-4px);}
			40%,60% {transform: translateX(4px);}
		}

		.inputbox::before {
			content: '';
			display: inline-block;
		}

		$(mapreduce(stylelabel, add_elements, 'a':'z'))

		.wordle-box:hover .inputbox {
			animation: fadeout 0s;
		}
		.wordle-box:hover .inputbox* {
			animation: fadeout 0s;
		}

		@keyframes rowbounce {
			0%,20% {transform: translateY(0)}
			40% {transform: translateY(-.33em)}
			50% {transform: translateY(0.05em)}
			60% {transform: translateY(-0.15em)}
			80% {transform: translateY(0.02em)}
			100% {transform: translateY(0)}
		}

		@keyframes flipin {
			0% {
		        transform: rotateX(0)
		    }
		
		    100% {
		        transform: rotateX(-90deg)
		    }
		}

		@keyframes flipout {
		    0% {
		        transform: rotateX(-90deg)
		    }
		
		    100% {
		        transform: rotateX(0)
		    }
		}

		$(mapreduce(add_elements, 0:2) do i
			"""
			.letter.feedback$i {
				background-color: $(colorlookup[i]);
			}
			"""
		end)

		$(mapreduce(add_elements, 0:2) do i
			"""
			.Key-module_key__kchQI.feedback$i {
				background-color: $(colorlookup[i]);
			}
			"""
		end)

		
		@keyframes addcolor0 {
			0% {background-color: rgba(0, 0, 0, 0); border: 2px solid rgb(86, 87, 88);}
			50% {background-color: rgba(0, 0, 0, 0); border: 2px solid rgb(86, 87, 88);}
			51% {background-color: $(colorlookup[0]); border: 0px solid rgba(0, 0, 0, 0);}
			100% {background-color: $(colorlookup[0]); border: 0px solid rgba(0, 0, 0, 0);}
		}
		@keyframes addcolor1 {
			0% {background-color: rgba(0, 0, 0, 0); border: 2px solid rgb(86, 87, 88);}
			50% {background-color: rgba(0, 0, 0, 0); border: 2px solid rgb(86, 87, 88);}
			51% {background-color: $(colorlookup[1]); border: 0px solid rgba(0, 0, 0, 0);}
			100% {background-color: $(colorlookup[1]); border: 0px solid rgba(0, 0, 0, 0);}
		}
		@keyframes addcolor2 {
			0% {background-color: rgba(0, 0, 0, 0); border: 2px solid rgb(86, 87, 88);}
			50% {background-color: rgba(0, 0, 0, 0); border: 2px solid rgb(86, 87, 88);}
			51% {background-color: $(colorlookup[2]); border: 0px solid rgba(0, 0, 0, 0);}
			100% {background-color: $(colorlookup[2]); border: 0px solid rgba(0, 0, 0, 0);}
		}

	.rejectmessage {
		position: absolute;
		margin: 0 auto;
		margin-top: calc(var(--container-width)*.01);
		width: var(--container-width);
		height: calc(var(--container-width)*6/5);
		display: flex;
		align-items: center;
		flex-direction: column;
	}

	.rejectmessage:empty {
		display: none;
	}
	
	.rejectmessage *::before {
		content: '';
		display: flex;
		background-color: rgba(255, 255, 255, 0.95);
		color: black;
		font-family: 'Arial';
		font-size: calc(var(--container-width)*.04);
		font-weight: bold;
		justify-content: center;
		align-items: center;
		border-radius: calc(var(--container-width)*.01);
		padding: calc(var(--container-width)*.035);
		margin: calc(var(--container-width)*.02);
		animation-name: fadeout;
		animation-duration: 300ms;
		animation-fill-mode: forwards;
		animation-delay: 1100ms;	
	}

	.rejectmessage .not-in-list::before {
		content: 'Not in word list';
	}

	.rejectmessage .short-guess::before {
		content: 'Not enough letters';
	}
	
	.rejectmessage .incomplete-feedback::before {
		content: 'Incomplete feedback';
	}

	.rejectmessage .submitted-feedback::before {
		content: 'Feedback submitted!';
		color: green;
		background-color: rgba(20, 20, 20, 0.5);
	}
	
	.rejectmessage .game-won::before {
		content: 'Congratulations!';
		color: green;
		background-color: rgba(20, 20, 20, 0.5);
	}

	.rejectmessage .game-lost::before {
		content: 'Better luck next time';
		color: red;
		background-color: rgba(20, 20, 20, 0.5);
	}
		}
	</style>
	"""
)

# ╔═╡ 286b4ce9-5412-4506-8334-983655e3ecba
const endgame_styles = HTML("""
<style>
	.gamewin::before {
		content: '';
		position: absolute;
		width: calc(var(--container-width));
		height: 100%;
		padding: 0;
		margin: 0;
		background-color: rgba(50, 50, 50, 0.8);
		animation: fadein 3s;
		
	}
	.gamewin::after {
		content: 'Game Won!';
		position: absolute;
		width: calc(var(--container-width));
		height: calc(var(--container-width)*6/5);
		padding: 0;
		margin: 0;
		font-family: "Arial", sans-serif;
		font-weight: bold;
		-webkit-font-smoothing: antialiased;
		text-transform: uppercase;
		font-size: calc(var(--container-width)/5/2.0); 
		color: white;
		display: flex;
		justify-content: center;
		align-items: center;
		animation: winmessage 3s;
	}
	@keyframes winmessage {
		0% {opacity: 0;}
		60% {opacity: 0; transform: scale(0.4);}
		80% {transform: scale(1.5);}
		100% {transform: scale(1.0);}
	}

	.gamewin:hover::after {
		animation: fadeout 1s forwards;
	}

	.gamewin:hover::before {
		animation: fadeout 1s forwards;
	}

	@keyframes fadein {
		0% {opacity: 0;}
	}

	@keyframes fadeout {
		100% {opacity: 0;}
	}

	@keyframes repeatwin {
		0% {transform: rotate(0deg);}
		25% {transform: rotate(180deg);}
		50% {transform: rotate(360deg);}
		75% {transform: rotate(540deg);}
		100% {transform: rotate(720deg);}
	}

	@keyframes winoverlay {
		0% {opacity: 0}
		60% {opacity: 0}
	}

	.gamelost::before {
		content: 'Hover to See Word';
		position: absolute;
		background-size: 100%;
		left: 0;
		top: 0;
		width: 100%;
		height: 100%;
		padding: 0;
		margin: 0;
		display: flex;
		justify-content: center;
		font-family: "Arial", sans-serif;
		font-weight: bold;
		-webkit-font-smoothing: antialiased;
		text-transform: uppercase;
		font-size: calc(var(--container-width)/5/5.0); 
		background-color: rgba(20, 20, 20, 0.5);
		border: 2px solid rgba(20, 20, 20, 0.5);
		animation: fadein 3s;
	}

	.gamelost:hover::before {
		animation: fadeout 0s forwards;
	}

	.gamelost::after {
		content: 'Game Lost :(';
		position: absolute;
		padding: calc(var(--container-width)/35);
		font-family: "Arial", sans-serif;
		font-weight: bold;
		-webkit-font-smoothing: antialiased;
		text-transform: uppercase;
		font-size: calc(var(--container-width)/5/2.0); 
		animation: losemessage 3s forwards;
	}

	@keyframes losemessage {
		0%,25% {color: rgba(0, 0, 0, 0); transform: translateX(calc(var(--container-width)/7)) translateY(10%);}
		50% {color: red; transform: translateX(calc(var(--container-width)/7)) translateY(100%);}
		75%, 100% {transform: translateX(calc(var(--container-width)/7)) translateY(50%);}
	}

	@keyframes loseoverlay {
		0% {opacity: 0; color: rgba(0, 0, 0, 0);}
		50% {opacity: 0; color: rgba(0, 0, 0, 0);}
		100% {opacity: 1;}
	}

	.wordle-game .gamelost:hover::after {
		content: attr(lose-message);
		font-family: "Arial", sans-serif;
		font-weight: bold;
		-webkit-font-smoothing: antialiased;
		text-transform: uppercase;
		font-size: calc(var(--container-width)/5/3.0); 
		color: green;
		text-shadow: 3px 3px black;
		display: flex;
		justify-content: center;
		align-items: center;
		background-color: rgba(20, 20, 20, 0.7);
		border: 2px solid rgba(0, 0, 0, 0);
		animation: showtext 1s forwards;
	}
	@keyframes showtext {
		0% {opacity: 0;}
		100% {opacity: 1; transform: translateX(50px);}
	}

	$(mapreduce(add_elements, 0:4) do i
	"""
	.box$i.win {
		animation: rowbounce 1000ms $(100*i + 600)ms;			
		}
	"""
	end)
</style>
""")

# ╔═╡ 80ab2d58-787a-4916-b2d1-2d56d8f212d9
const inputstyle = HTML("""
	<style>		
	.inputbox.anim {
			animation: addletter 100ms forwards;
		}
	@keyframes addletter {
		0% {transform: scale(0.8); opacity: 0;}
		40% {transform: scale(1.1); opacity: 1;}
		100% {border-color: rgb(86, 87, 88);}
	}
	</style>
""")

# ╔═╡ d7c69c40-5ae5-4f95-bff6-c247cb2d1c33
function wordle_restyle(f::Real, class::String) 
	"""
	<style>
		.wordle-box.$class {
			width: calc($f*var(--container-width));
			height: calc($f*var(--container-width)/5);
			grid-gap: calc($f*var(--container-width)/60);
			margin: calc($f*var(--container-width)/15);
		}
		.wordle-box.$class * {
			font-size: calc($f*var(--container-width)/5/2.0); 
		}
	</style>
	"""
end

# ╔═╡ 704d5d28-3ded-4d1a-a89e-4e636b88cd05
const fliptime = "250ms"

# ╔═╡ 8d9cb914-1f6b-4ed0-b84b-783727a8c3e8
function makeanimationclass(fval::Integer, i::Integer, iswin::Bool)
	delaytime = 100*i
	if iswin
		"""
		.box$i.win {
			animation: 	flipin $fliptime $(delaytime)ms ease-in, 
						flipout $fliptime $(delaytime+250)ms ease-in,
						addcolor2 500ms $(delaytime)ms forwards,
						rowbounce 1000ms $(delaytime+600)ms;			
			}
		"""
	else
		"""
		.box$i.feedback$fval,.box$i[feedback="$fval"] {
			animation: 	flipin $fliptime $(delaytime)ms ease-in, 
						flipout $fliptime $(delaytime+250)ms ease-in,
						addcolor$fval 500ms $(delaytime)ms ease-in forwards;
		}
		"""
	end
end

# ╔═╡ 501e5240-8d3d-4db4-b053-1c4148cd128f
# create animation classes for every possible square
const boxanimations = HTML("""
<style>
$(mapreduce(a -> makeanimationclass(a...), add_elements, ((fval, i, iswin) for fval in [0, 1, 2] for i in 0:4 for iswin in [true, false])))
</style>
""")

# ╔═╡ b3bd847c-f21f-425c-ae05-abd95da6e858
show_pattern(pnum::Integer; kwargs...) = show_pattern(digits(pnum, base=3, pad=5); kwargs...)

# ╔═╡ 7804e77c-f525-4ecd-a91e-fbd9d6ccf22a
const winfeedback = fill(EXACT, 5)

# ╔═╡ 0e73c1ad-89b1-40d3-842e-398c98ecce14
function show_pattern(feedback::AbstractVector{T}; boxcontent = i -> "", sizepct = 1.0, repeat = 1) where T <: Integer
	colors = [colorlookup[i] for i in feedback]

	classname = string("a", hash(feedback), hash(boxcontent))

	function make_box(i)
		"""
		<div class = "inputbox box$(i-1) feedback$(feedback[i])" label = "$(lowercase(boxcontent(i)))"></div>
		"""
	end

	make_win_box(i) = """<div class = "inputbox box$(i-1) win" label = "$(lowercase(boxcontent(i)))"></div>"""

	f = all(feedback .== winfeedback) ? make_win_box : make_box
	
	restyle = if sizepct == 1
		""""""
	else
		wordle_restyle(sizepct, classname)
	end

	HTML("""
	<div class="wordle-box $classname">
		$(mapreduce(f, add_elements, 1:5))
	</div>
	$restyle
	""")
end

# ╔═╡ d4a07d83-cab3-4ca3-bed0-3880ade5ff6a
show_pattern(example_feedback)

# ╔═╡ 8353133e-7559-48fe-b1b5-51b330e70182
HTML("""
<div style = "display: grid; align-content: center; justify-items: center; grid-template-columns: repeat(10, 1fr); grid-template-rows: repeat(10, 1fr);">
$(mapreduce(f -> show_pattern(f; sizepct = 0.1), add_elements, feedback_matrix[1:10, 1:10]))
</div>
""")

# ╔═╡ 64014b10-9974-475a-8625-fef445fe8e4f
md"""
## Single Guess Input Input Element
"""

# ╔═╡ e426b806-4435-49b9-824a-04fe34a48e9e
begin
	struct WordleInput{T <: AbstractString}
		word::T
		addindex::Integer
	end

	WordleInput(;default="") = WordleInput(uppercase(default), length(default)-1)
	
	function Bonds.show(io::IO, m::MIME"text/html", input::WordleInput)
		show(io, m, HTML("""
		<span>
		<input class=wordleinput type=text oninput="this.value = this.value.replace(/[^a-zA-Z]/, '')" maxlength=5 $(isempty(input.word) ? "" : "value=$(input.word)") size=7>
		<style>
			.wordleinput {
				text-transform:uppercase;
				font-family: "Arial", sans-serif;
				font-weight: bold;
				font-size: calc(var(--container-width)/5/2.0); 
			}
		</style>
		<script>
			const span = currentScript.parentElement;
			const inputbox = span.querySelector(".wordleinput");
			span.value = [inputbox.value, inputbox.value.length-1];
			inputbox.addEventListener('keydown', handleWordleInput);
			inputbox.addEventListener('input', handleInput); 
			
			function handleInput(e) {
				span.value[0] = inputbox.value;
				if (inputbox.value.length === 0) {
					span.value[1] = -1;
				}
			}
		
			function handleWordleInput(e) {
				if (e.keyCode === 8) {
					span.value[1] = -1;
				} else if (inputbox.value.length === 5) {
					span.value[1] = -1;
				} else if (e.keyCode >= 65 && e.keyCode <= 90) {
					span.value[1] = inputbox.value.length;
				} else {
					span.value[1] = -1;
				} 
			}
		</script>
		</span>
		"""))
	end

	Base.get(input::WordleInput) = input
	Bonds.initial_value(input::WordleInput) = (word = input.word, addindex = input.addindex)
	Bonds.possible_values(input::WordleInput) = Bonds.InfinitePossibilities()
	Bonds.transform_value(input::WordleInput, val_from_js) = (word=uppercase(val_from_js[1]), addindex=val_from_js[2])
end

# ╔═╡ 10eb71ed-df57-49b2-ae98-336191346b29
md"""
## Playable Game Element
"""

# ╔═╡ bab1c451-bf1b-495c-9448-0be284063733
mapreduce(add_elements, 1:5) do nrows
	"""
	.wordle-game-input.maxrows$nrows .wordle-game-grid {
		height: calc(var(--container-width)*$nrows/5);
	}
	"""
end |> add_style

# ╔═╡ ccca515b-17ad-4469-ac57-61c2f3f86e62
add_style("""

	.inputbox.feedback0.locked {
		animation: flipin 250ms ease-in, flipout 250ms 250ms ease-in, addcolor0 500ms ease-in forwards, addborder 250ms 250ms forwards;
	}

	.inputbox.feedback1.locked {
		animation: flipin 250ms ease-in, flipout 250ms 250ms ease-in, addcolor1 500ms ease-in forwards, addborder 250ms 250ms forwards;
	}

	.inputbox.feedback2.locked {
		animation: flipin 250ms ease-in, flipout 250ms 250ms ease-in, addcolor2 500ms ease-in forwards, addborder 250ms 250ms forwards;
	}



	.inputrow {
		display: grid;
		grid-template-columns: repeat(5, 1fr);
		grid-gap: 5px;
		width: calc(var(--container-width));
	}

	.inputrow.locked {
		//background-color: rgb(40, 40, 40);
		//border: 3mm ridge rgba(100, 200, 100, .4);
		//box-shadow: 3px 3px forestgreen;
		animation: addborder 500ms forwards;
	}

	@keyframes addborder {
		100% {background-color: rgb(40, 40, 40); border: 3mm ridge rgba(100, 200, 100, .4);}
	}

	.wordle-game-rows {
		display: grid;
		grid-gap: 5px;
		padding: 10px;
	}
	
""")

# ╔═╡ 67e57ae3-9834-43e6-ad46-91a6c046b69b
md"""
## Game Input Element
"""

# ╔═╡ 1baafa2c-b087-41be-9e5f-6adea8a985fc
begin
	struct WordleGameInput{T <: Integer, S <: AbstractString}
		guessnum::T
		guesses::Vector{S}
		feedback::Vector{Vector{T}}
		numguesses::T
		showletters::Bool
		possible_answers::Vector{S}
		status::Symbol
		undos::Vector{Tuple{Vector{S}, Vector{T}}}
	end
	
	WordleGameInput(;nguesses = 6, showletters=true) = WordleGameInput(0, Vector{String}(), Vector{Vector{Int64}}(), nguesses, showletters, nyt_valid_words, :active, Vector{Tuple{Vector{String}, Vector{Int64}}}())

	Base.get(input::WordleGameInput) = Bonds.initial_value(input)
	Bonds.initial_value(input::WordleGameInput) = (guesses = Vector{String}(), feedback = Vector{Vector{Int64}}(), numguesses = input.numguesses, status = :active, undoes = input.undos)
	Bonds.possible_values(input::WordleGameInput) = Bonds.InfinitePossibilities()
	Bonds.transform_value(input::WordleGameInput, val_from_js) = (guesses = Vector{String}(val_from_js[1]), feedback = Vector{Vector{Int64}}(val_from_js[2]), numguesses = input.numguesses, status = Symbol(val_from_js[3]), undos = isempty(val_from_js[4]) ? input.undos : collect(zip(val_from_js[4], val_from_js[5])))
	
	function Bonds.show(io::IO, m::MIME"text/html", game::WordleGameInput)
		nrows = game.numguesses

		lettergrid = if game.showletters
			nyt_keyboard
		else
			""""""
		end

		makerow(r) = mapreduce(c -> """<div class = "inputbox box0" row = "$r" tabindex=0></div>""", add_elements, 0:4)
		
		show(io, m, HTML("""
		<span class = "wordle-game-input maxrows$nrows" tabindex=0>
		<div class = "buttons">
		<button class=resetgame>Reset</button>
		<button class=undoguess>Undo Guess</button>
		<button class=clearundo>Clear Undos</button>
		</div>
		
		<div class = wordle-game-rows>
			$(mapreduce(r -> 
				"""
				<div class = "inputrow row$r">
					$(makerow(r))
				</div>
				"""
				, add_elements, 0:nrows-1))
		</div>
		<div class = rejectmessage></div>
		$lettergrid
		
		<script>
			const letters = [...Array(26).keys()].map((n) => String.fromCharCode(97 + n));
		
			const keyLookup = {"0":48,"1":49,"2":50,"3":51,"4":52,"5":53,"6":54,"7":55,"8":56,"9":57,"d":68,"b":66,"a":65,"s":83,"i":73,"f":70,"k":75,"ß":219,"Dead":220,"+":187,"ü":186,"p":80,"o":79,"u":85,"z":90,"t":84,"r":82,"e":69,"w":87,"g":71,"h":72,"j":74,"l":76,"ö":192,"ä":222,"#":191,"q":81,"y":89,"x":88,"c":67,"v":86,"n":78,"m":77,",":188,".":190,"-":189,"ArrowRight":39,"ArrowLeft":37,"ArrowUp":38,"ArrowDown":40,"PageDown":34,"Clear":12,"Home":36,"PageUp":33,"End":35,"←":8,"↵":13};

			//const keyLookup = {"0":48, "1":49};
			const span = currentScript.parentElement;
			const reset = span.querySelector(".resetgame");
			const undo = span.querySelector(".undoguess");
			const clearUndosButton = span.querySelector(".clearundo");
			reset.addEventListener("click", resetGame);
			undo.addEventListener("click", undoGuess);
			clearUndosButton.addEventListener("click", clearUndos);
		
			const game = span.querySelector(".wordle-game-grid");
			const messageDisplay = span.querySelector(".rejectmessage");
			const validWords = $(game.possible_answers);
			const validWordSet = new Set(validWords);
			const rowElements = [...span.querySelectorAll(".inputrow")];
		
			let answerIndex = 0;
			const gameContainer = span.querySelector(".wordle-game-input");
			const letterButtons = [...span.querySelectorAll(".Key-module_key__kchQI")];
			letterButtons.map(elem => elem.addEventListener("mousedown", handleKeyClick));

			function handleKeyClick(e) {
				// console.log(e.target.getAttribute('data-key'));	
				let key = e.target.getAttribute('data-key');
				processKeyCode(keyLookup[key], key);
			}
		
			span.addEventListener("keydown", handleKeyDown);
			span.addEventListener("keyup", handleKeyUp);
			let col = -1;
			let row = 0;
			span.value = [[], [], 'active', [], []];
			let rows = $(collect(0:nrows-1));
			const rowElems = rows.map(row => [...span.querySelectorAll('.inputbox[row="'+row+'"]')]);

			const inputBoxes = [...span.querySelectorAll(".inputbox")];
			inputBoxes.map(e => e.addEventListener("click", feedbackClick));

			function toggleFeedback(classes, num, time) {
				setTimeout(()=>classes.add("feedback"+num), time);
			}
		
			function feedbackClick(e) {
				let elem = e.target
				let r = elem.getAttribute("row");
				let rowParent = span.querySelector('.inputrow.row'+r);
				if (e.target.hasAttribute("label") && !rowParent.classList.contains("locked")) {
					let newf = 0;
					if (elem.hasAttribute("feedback")) {
						let f = elem.getAttribute("feedback");
						newf = parseInt(f) + 1;
						if (newf == 3) {newf = 0;}
					}
					elem.classList.remove("anim");
					elem.removeAttribute('feedback');
					setTimeout(() => elem.setAttribute('feedback', newf), 100);
				}
			}

			function checkFeedback(elem) {
				let v = [0, 1, 2];
				//console.log(elem.classList);
				let out = v.some(v => elem.classList.contains('feedback'+v));
				//console.log('check feedback is '+out);
				return out;
			}

			function getFeedback(elem) {
				let classes = elem.classList;
				let f = -1
				if (classes.contains('feedback0')) {
					f = 0;
				} else if (classes.contains('feedback1')) {
					f = 1;
				} else if (classes.contains('feedback2')) {
					f = 2;
				}
				return f;
			}

			function getRowFeedback(elems) {
				elems.map(e => getFeedback(e));
			}
			

			function getKeySelector(e) {
				let k = e.key; 
				if (e.key == "Backspace") {
					k = "←";
				} else if (e.key == "Enter") {
					k = "↵";
				}
							
				let selector = '.Key-module_key__kchQI[data-key="' + k + '"]'
				// console.log(selector);
				return selector;
			}

			function handleKeyDown(e) {
				let selector = getKeySelector(e);
				span.querySelector(selector).classList.add('pressed');
				processKeyCode(e.keyCode, e.key);
			}

			function handleKeyUp(e) {
				let selector = getKeySelector(e);
				span.querySelector(selector).classList.remove('pressed');
			}

			function removeClasses(elem) {
				let classes = ["anim", "feedback0", "feedback1", "feedback2"];
				classes.map(c => elem.classList.remove(c));
			}

			function lockInput(elem) {
				elem.classList.add("locked");
			}
		
			function processKeyCode(code, key) {
				
				if (span.value[3] == 'loss') {
					console.log("game lost");
					showMessage("game-lost");
				}
				else if (span.value[3] == 'win') {
					console.log("game won");
					showMessage("game-won");
				}
				else {
					let elems = rowElems[row];
					if (code >= 65 && code <= 90) {
						col += 1;
						if (col > 4) {
							col = 4;
						}
						elems[col].classList.add("anim");
						elems[col].setAttribute('label', key);
					} else if (code == 8) {
						if (col > -1) {
							elems[col].removeAttribute('label');
							elems[col].removeAttribute('feedback');
							removeClasses(elems[col]);
						}
						col -= 1;
						if (col < -1) {
							col = -1;
						}
					} else if (code == 13) { 
						let fullWord = elems.every(e => e.hasAttribute("label"))
						let fullFeedback = elems.every(e => e.hasAttribute('feedback'));
						if (col == 4 && fullWord && fullFeedback) {
							var currentWord = elems.map(e => e.getAttribute('label')).reduce((a, b) => a+b);
							if (validWordSet.has(currentWord)) {
								span.value[0][row] = currentWord;				
								col = -1;
								var feedback = elems.map(e => parseInt(e.getAttribute('feedback')));
								span.value[1][row] = feedback;
								span.dispatchEvent(new CustomEvent('input'));
								showMessage("submitted-feedback");
								// elems.map(e => lockInput(e));
								rowElements[row].classList.add("locked");
								applyFeedback(feedback, elems);
								row += 1;
							} else {
								let rmClasses = ['anim', 'feedback0', 'feedback1', 'feedback2']
								elems.map(e => {
									rmClasses.map(c => e.classList.remove(c));
									e.classList.add("invalid");
									e.removeAttribute('feedback');
									setTimeout(removeInvalid, 600, e);
								})
								showMessage("not-in-list");
								console.log(currentWord, "is an invalid guess");
							}	
						} else {
							if (!fullWord) {
								showMessage("short-guess");						
								console.log("Guess too short");
							} else if (!fullFeedback) {
								showMessage("incomplete-feedback");
								console.log("Incomplete feedback");
							}
		
							for (let i = 0; i<5; i++) {
								let e = elems[i]
								e.classList.remove("anim");
								e.classList.add("invalid");
								setTimeout(removeInvalid, 600, e);
							}
							
						}
					}
				}
			}
			function removeInvalid(e) {
				e.classList.remove("invalid");
			}

			function showMessage(msgClass) {
				var message = document.createElement("div");
				message.classList.add(msgClass);
				let children = [...messageDisplay.childNodes];
				messageDisplay.insertBefore(message, children[0]);
				setTimeout(removeMessage, 1500, message);
			}

			function removeMessage(c) {
				messageDisplay.removeChild(c);
				if (!messageDisplay.hasChildNodes()) {
					messageDisplay.classList.remove("displaymessage");
					messageDisplay.classList.remove("short-guess");
				}
			}
		
			function resetClasses() {
				col = -1;
				row = 0;
				inputBoxes.map(elem => {
					elem.removeAttribute("label");
					let classes = ["anim", "invalid", "win"]
					classes.map(c => elem.classList.remove(c));
					elem.removeAttribute('feedback');
				})
				for (let i = 0; i < letterButtons.length; i++) {
					removeColors(letterButtons[i]);
				}
				rowElements.map(e => e.classList.remove("locked"));
			}

			function undoGuess() {
				if (row > 0) {
					col = -1;
					rowElements[row-1].classList.remove("locked");
					
					let boxes1 = [...span.querySelectorAll('.inputbox[row="'+(row-1)+'"]')];
					let boxes2 = [...span.querySelectorAll('.inputbox[row="'+row+'"]')];
					
					function clearBoxes(boxes) {
						boxes.map(elem => {
							elem.removeAttribute("label");
							let classes = ["anim", "invalid", "win"]
							classes.map(c => elem.classList.remove(c));
							elem.removeAttribute('feedback');
						})
					}
					clearBoxes(boxes1);
					clearBoxes(boxes2);
					// transfer guess and feedback to undo lists
					span.value[3].push(span.value[0].pop())
					span.value[4].push(span.value[1].pop())
					span.dispatchEvent(new CustomEvent('input'));
					row -= 1;
				}
			}

			function clearUndos() {
				span.value[3] = [];
				span.value[4] = [];
				span.dispatchEvent(new CustomEvent('input'));
			}
			
		
			function removeColors(item) {
				item.classList.remove("feedback0");
				item.classList.remove("feedback1");
				item.classList.remove("feedback2");
			}
		
			function resetGame() {
				resetClasses();
				span.value[0] = [];
				span.value[1] = [];
				span.value[2] = 'active';
				span.value[3] = [];
				span.value[4] = [];
				span.dispatchEvent(new CustomEvent('input'));
				reset.blur();
				span.focus();
			}

			function applyFeedback(feedback, elems) {
				console.log('Feedback is ' + feedback);
				elems.map((e, index) => {
					e.classList.remove('anim');
					let letter = e.getAttribute("label");
					span.querySelector('.Key-module_key__kchQI[data-key="'+letter+'"]').classList.add('feedback'+feedback[index]);
				});
			}
			
		</script>
		</span>
		"""))
	end
	#add event listeners for keyboard keys and change layout to querty
end

# ╔═╡ b9beca61-7efc-4186-81c2-1ceda103c801
begin
	struct WordleGame{T <: Integer, S <: AbstractString}
		guessnum::T
		guesses::Vector{S}
		feedback::Vector{Vector{T}}
		answer_index_list::Vector{T}
		answerindex::T
		numguesses::T
		showletters::Bool
		possible_answers::Vector{S}
		status::Symbol
		undos::Vector{Tuple{Vector{S}, Vector{T}}}
	end

	function sample_random_answers(n)
		future_answers = [sample_answer() for _ in 1:n]
		[word_index[a] for a in future_answers]
	end
	
	WordleGame(;answer_index_list=sample_random_answers(5000), nguesses = 6, showletters=true) = WordleGame(0, Vector{String}(), Vector{Vector{Int64}}(), answer_index_list, first(answer_index_list), nguesses, showletters, nyt_valid_words, :active, Vector{Tuple{Vector{String}, Vector{Int64}}}())

	WordleGame(answer::AbstractString; kwargs...) = WordleGame(;answer_index_list = [word_index[lowercase(answer)]], kwargs...)

	Base.get(input::WordleGame) = Bonds.initial_value(input)
	Bonds.initial_value(input::WordleGame) = (guesses = Vector{String}(), feedback = Vector{Vector{Int64}}(), answerindex = input.answerindex, numguesses = input.numguesses, status = :active, undos = input.undos)
	Bonds.possible_values(input::WordleGame) = Bonds.InfinitePossibilities()
	Bonds.transform_value(input::WordleGame, val_from_js) = (guesses = Vector{String}(val_from_js[1]), feedback = Vector{Vector{Int64}}(val_from_js[2]), answerindex = val_from_js[3], numguesses = input.numguesses, status = Symbol(val_from_js[4]), undos = input.undos)

	function makelettersquare(c::Char)
		"""
		<div class="letter $c">$c</div>
		"""
	end
	
	function Bonds.show(io::IO, m::MIME"text/html", game::WordleGame)
		wordindex = game.answerindex
		answer = game.possible_answers[wordindex]
		future_answer_index = game.answer_index_list
		nrows = game.numguesses
		gameclass = "answer-number-$(game.answerindex)"
		gameindex = "gameindex-$(hash(game))"

		resizegrid = if nrows != 6
			"""
			.wordle-game.$gameindex .wordle-game-grid {
				height: calc(var(--container-width)*$nrows/5);
			}
			"""
		else
			""""""
		end

		lettergrid = if game.showletters
			nyt_keyboard
		else
			""""""
		end
		
		show(io, m, HTML("""
		<span class = "wordle-game $gameclass $gameindex" tabindex=0>
		<div class = "buttons">
		<button class=resetgame>Reset</button>
		<button class=newGame>New Game</button>
		</div>
		
		<div class = wordle-game-grid>
			$(mapreduce(a -> """<div class = "inputbox row$(a[1]) box$(a[2])"></div>""", add_elements, ((r, c) for r in 0:nrows-1 for c in 0:4)))
		</div>
		<div class = rejectmessage></div>
		$lettergrid
		
		<style>
		$resizegrid
		</style>
		<script>
			const letters = [...Array(26).keys()].map((n) => String.fromCharCode(97 + n));
		
			const keyLookup = {"0":48,"1":49,"2":50,"3":51,"4":52,"5":53,"6":54,"7":55,"8":56,"9":57,"d":68,"b":66,"a":65,"s":83,"i":73,"f":70,"k":75,"ß":219,"Dead":220,"+":187,"ü":186,"p":80,"o":79,"u":85,"z":90,"t":84,"r":82,"e":69,"w":87,"g":71,"h":72,"j":74,"l":76,"ö":192,"ä":222,"#":191,"q":81,"y":89,"x":88,"c":67,"v":86,"n":78,"m":77,",":188,".":190,"-":189,"ArrowRight":39,"ArrowLeft":37,"ArrowUp":38,"ArrowDown":40,"PageDown":34,"Clear":12,"Home":36,"PageUp":33,"End":35,"←":8,"↵":13};

			//const keyLookup = {"0":48, "1":49};
		
			const reset = document.querySelector(".wordle-game.$gameclass .resetgame");
			const newGame =  document.querySelector(".wordle-game.$gameclass .newGame");
			reset.addEventListener("click", resetGame);
			newGame.addEventListener("click", makeNewGame);
			const span = currentScript.parentElement;
			const game = document.querySelector(".wordle-game.$gameclass .wordle-game-grid");
			const messageDisplay = document.querySelector(".wordle-game.$gameclass .rejectmessage");
			const futureAnswerIndex = $future_answer_index;
			const validWords = $(game.possible_answers);
			const validWordSet = new Set(validWords);
		
			let answerIndex = 0;
			const gameContainer = document.querySelector(".wordle-game.$gameclass");
			const letterButtons = [...document.querySelectorAll(".wordle-game.$gameclass .Key-module_key__kchQI")];
			letterButtons.map(elem => elem.addEventListener("mousedown", handleKeyClick));

			game.setAttribute('lose-message', 'Word was '+validWords[futureAnswerIndex[answerIndex]-1]);

			function handleKeyClick(e) {
				// console.log(e.target.getAttribute('data-key'));	
				let key = e.target.getAttribute('data-key');
				processKeyCode(keyLookup[key], key);
			}
		
			span.addEventListener("keydown", handleKeyDown);
			span.addEventListener("keyup", handleKeyUp);
			let col = -1;
			let row = 0;
			span.value = [[], [], $wordindex, 'active'];
			let rows = $(collect(0:nrows-1));
			const rowElems = rows.map(row => [...document.querySelectorAll(".wordle-game.$gameclass .inputbox.row"+row)]);

			function getKeySelector(e) {
				let k = e.key; 
				if (e.key == "Backspace") {
					k = "←";
				} else if (e.key == "Enter") {
					k = "↵";
				}
							
				let selector = '.wordle-game.$gameindex .Key-module_key__kchQI[data-key="' + k + '"]'
				// console.log(selector);
				return selector;
			}

			function handleKeyDown(e) {
				let selector = getKeySelector(e);
				document.querySelector(selector).classList.add('pressed');
				processKeyCode(e.keyCode, e.key);
			}

			function handleKeyUp(e) {
				let selector = getKeySelector(e);
				document.querySelector(selector).classList.remove('pressed');
			}
		
			function processKeyCode(code, key) {
				
				if (span.value[3] == 'loss') {
					console.log("game lost");
					showMessage("game-lost");
				}
				else if (span.value[3] == 'win') {
					console.log("game won");
					showMessage("game-won");
				}
				else {
					let elems = rowElems[row];
					if (code >= 65 && code <= 90) {
						col += 1;
						if (col > 4) {
							col = 4;
						}
						elems[col].classList.add("anim");
						elems[col].setAttribute('label', key);
						} else if (code == 8) {
						if (col > -1) {
							elems[col].removeAttribute("label");
							elems[col].classList.remove("anim");
						}
						col -= 1;
						if (col < -1) {
							col = -1;
						}
					} else if (code == 13) { 
			
						if (col == 4 && elems[4].hasAttribute("label")) {
							var currentWord = elems.map(e => e.getAttribute('label')).reduce((a, b) => a+b);
							if (validWordSet.has(currentWord)) {
								span.value[0][row] = currentWord;				
								col = -1;
								var answer = validWords[futureAnswerIndex[answerIndex]-1];
								var feedback = getFeedback(currentWord, answer);
								span.value[1][row] = feedback;
								applyFeedback(feedback, elems);
								span.dispatchEvent(new CustomEvent('input'));
								row += 1;
							} else {
								
								for (let i = 0; i<5; i++) {
									let e = elems[i]
									e.classList.remove("anim");
									e.classList.add("invalid");
									setTimeout(removeInvalid, 600, e);
								}
								showMessage("not-in-list");
								console.log(currentWord, "is an invalid guess");
							}	
						} else {
							
								for (let i = 0; i<5; i++) {
									let e = elems[i]
									e.classList.remove("anim");
									e.classList.add("invalid");
									setTimeout(removeInvalid, 600, e);
								}
								showMessage("short-guess");						
								console.log("Guess too short");
						}
					}
				}
			}
			function removeInvalid(e) {
				e.classList.remove("invalid");
			}

			function showMessage(msgClass) {
				var message = document.createElement("div");
				message.classList.add(msgClass);
				let children = [...messageDisplay.childNodes];
				messageDisplay.insertBefore(message, children[0]);
				setTimeout(removeMessage, 1500, message);
			}

			function removeMessage(c) {
				messageDisplay.removeChild(c);
				if (!messageDisplay.hasChildNodes()) {
					messageDisplay.classList.remove("displaymessage");
					messageDisplay.classList.remove("short-guess");
				}
			}
		
			function resetClasses() {
				col = -1;
				row = 0;
				for (const child of game.children) {
					child.removeAttribute("label");
					child.classList.remove("anim");
					child.classList.remove("invalid");
					child.classList.remove("feedback0");
					child.classList.remove("feedback1");
					child.classList.remove("feedback2");
					child.classList.remove("win");
				}
				game.classList.remove("gamewon");
				game.classList.remove("gamelost");
				game.classList.remove("gameover");
				for (let i = 0; i < letterButtons.length; i++) {
					removeColors(letterButtons[i]);
				}
			}
		
			function removeColors(item) {
				item.classList.remove("feedback0");
				item.classList.remove("feedback1");
				item.classList.remove("feedback2");
			}
		
			function resetGame() {
				resetClasses();
				span.value[0] = [];
				span.value[1] = [];
				span.value[3] = 'active';
				span.dispatchEvent(new CustomEvent('input'));
				reset.blur();
				span.focus();
			}
		
			function makeNewGame() {
				resetClasses();
				span.value[0] = [];
				span.value[1] = [];
				span.value[3] = 'active';
				answerIndex += 1;
				if (answerIndex > futureAnswerIndex.length-1) {
					answerIndex = 0;
				}
				span.value[2] = futureAnswerIndex[answerIndex];
				span.dispatchEvent(new CustomEvent('input'));
				newGame.blur();
				span.focus();
				game.setAttribute('lose-message', 'Word was '+validWords[futureAnswerIndex[answerIndex]-1]);
			}

			function getFeedback(guess, answer) {
				let feedback = new Uint8Array(5);
				//initialize dictionary of letter counts from answer
				let letterCounts = Object.fromEntries([0, 1, 2, 3, 4].map(i => [answer[i], 0]));
		
				//green pass
				for (let i = 0; i < guess.length; i++) {
					let l = answer[i];
					letterCounts[l] += 1;
					if (guess[i] == l) {
						feedback[i] = 2;
						letterCounts[l] -= 1;
					}
				}
				//yellow pass
				for (let i = 0; i < guess.length; i++) {
					let l = guess[i];
					if (feedback[i] == 0 && letterCounts[l] > 0) {
						feedback[i] = 1;
						letterCounts[l] -= 1;
					}
				}
				return feedback;
			}

			function applyFeedback(feedback, elems) {
				console.log('Feedback is ' + feedback);
				function addLabel(e, index) {
					e.classList.remove('anim');
					let letter = e.getAttribute("label");
					span.querySelector('.Key-module_key__kchQI[data-key="'+letter+'"]').classList.add('feedback'+feedback[index]);
					e.classList.add('feedback'+feedback[index]);
				}
				elems.map((e, index) => {setTimeout(() => {addLabel(e, index);}, index*200);});
				if (feedback.every(f => {return f == 2})) {
					console.log('game won');
					setTimeout(() => showMessage("game-won"), 1900);
					elems.map((e, index) => setTimeout(() => {
						e.classList.remove('feedback2');
						e.classList.add('win');
					}, 1400));
					setTimeout(()=>{game.classList.add("gamewon")}, 2000);
					span.value[3] = 'win';
				} else if (row == $(nrows-1)) {
					console.log('game lost');
					setTimeout(() => showMessage("game-lost"), 1900);
					setTimeout(() => {game.classList.add("gamelost")}, 1500);
					span.value[3] = 'loss';
				} else {
					feedback.map((f, index) => { 
						setTimeout(()=>{elems[index].classList.add('feedback'+f)}, index*100)});
					if (row == $(nrows-1)) {
						console.log('game lost');
						showMessage("game-lost");
						setTimeout(() => {game.classList.add("gamelost")}, 1500);
						span.value[3] = 'loss';
					}
				}
			}
			
		</script>
		</span>
		"""))
	end
	#add event listeners for keyboard keys and change layout to querty
end

# ╔═╡ e460fde6-f65a-4af2-9517-327a187b112a
@bind rawanswer confirm(WordleInput(default="apple"))

# ╔═╡ 8a090c51-aa61-4090-96a1-8f4833bb9983
WordleGame(rawanswer.word; nguesses = 1)

# ╔═╡ 70dcac79-2839-4c1d-8f48-cbe294b3ffb0
@bind example_state WordleGameInput()

# ╔═╡ bf1c13cd-a066-4865-b43f-0cba3d3df8d2
md"""Proposed Guess: $(@bind example_new_guess Select(nyt_valid_words))"""

# ╔═╡ 69aacb14-52b2-49a2-a2bc-e8c07b9e1986
md"""Adjust Transition State Size: $(@bind size_adj confirm(Slider(100:500, default = 100)))"""

# ╔═╡ 83302fc6-49ff-4d25-9fd7-0b491b79fc73
md"""
Number of Guesses to Evaluate $(@bind num_guess_evals NumberField(1:length(nyt_valid_inds), default = 10))
"""

# ╔═╡ 01be89b2-6233-4e11-b142-fa6ad05880a3
begin
	num_guess_evals
	@bind run_policy_iteration CounterButton("Run policy iteration")
end

# ╔═╡ 57281f5c-4c93-4308-b1e7-e5a890655262
if run_policy_iteration > 0
	evaluate_wordle_state(WordleState(); num_evaluations = num_guess_evals)
else
	md"""Waiting to run policy evaluation for the top $num_guess_evals guesses"""
end

# ╔═╡ 9f9b8db2-7c74-49f0-b067-bb6aa433fbe0
md"""Show Only Hard Mode Guesses: $(@bind filter_hard CheckBox())"""

# ╔═╡ faade165-8066-4deb-93a4-2b7daabd57dc
evaluate_wordle_state(new_test_state; num_evaluations = 100, filter_hard = filter_hard)

# ╔═╡ caaeeaee-94b7-498d-a746-a2c5c7177347
@bind mcts_reset Button("Click to reset MCTS Evaluation")

# ╔═╡ fb8439a3-ee95-487f-a755-ffcbd8b3c381
begin 
	mcts_reset
	@bind stop_mcts_eval CounterButton("Click to stop MCTS Wordle Evaluation")
end

# ╔═╡ d9d54e56-a32f-4ffb-820d-df0b7918c78c
begin
	mcts_reset
	@bind mcts_counter CounterButton("Click to run MCTS Wordle Evaluation")
end

# ╔═╡ 7cd9cb06-4731-4eaa-b745-32118278d360
if stop_mcts_eval > 0
	md"""
	Evaluation stopped.  Wait for loop to end below and then press reset before running
	"""
elseif mcts_counter > 0
	md"""
	Evaluation started
	"""
else
	md"""
	Waiting to run mcts evaluation
	"""
end

# ╔═╡ 41e0cddb-e78d-477b-849a-124754340a3c
if mcts_counter > 0
	@use_effect([]) do
		t = time()
		schedule(Task() do
			nruns = 1_000
			nsims = 10
			for i in 1:nruns
				stop_mcts_eval > 0 && break	
				elapsed_minutes = (time() - t)/60
				etr = (elapsed_minutes * nruns / i) - elapsed_minutes
				set_run("Running $i of $nruns after $(round(Int64, (time() - t)/60)) minutes.  Estimated $(round(Int64, etr)) minutes left")
				output = @spawn show_afterstate_mcts_guesses(run_wordle_afterstate_mcts(WordleState(), nsims; tree_values = tree_values, sim_message = false, p_scale = 100f0), WordleState())
				set_mcts_options(fetch(output))
				sleep(.01)
			end
			if stop_mcts_eval > 0
				set_run("Interrupted")
			else
				set_run("Completed after $(round(Int64, (time() - t) / 60)) minutes")
			end
		end)
	end
end

# ╔═╡ d7b0a4ba-ba18-41ad-ade3-dde119f08a13
md"""
## Display Games
"""

# ╔═╡ 7e8be36e-5237-49f6-95ec-b9cb24b32c34
begin
	function show_wordle_game(answer::AbstractString, guesses::AbstractVector{T}; kwargs...) where T <: AbstractString
		#check that all guesses and answer are in valid word list
		@assert haskey(word_index, answer)
		@assert all(g -> haskey(word_index, g), guesses)
		
		#calculate feedback for each guess
		feedbacklist = [convert_bytes(get_feedback(guess, answer)) for guess in guesses]
		# gamelist = zip(guesses, feedbacklist)
		show_wordle_game(guesses, feedbacklist; kwargs...)
	end

	show_wordle_game(s::WordleState; kwargs...) = show_wordle_game(s.guess_list, s.feedback_list; kwargs...)
	show_wordle_game(s::WordleState{0}; kwargs...) = show_wordle_game(Vector{String}(), s.feedback_list; kwargs...)

	show_wordle_game(guess_list::AbstractVector{T}, feedbacklist; kwargs...) where T<:Integer = show_wordle_game([nyt_valid_words[i] for i in guess_list], feedbacklist; kwargs...)

	function show_wordle_game(guesses::AbstractVector{S}, feedbackints::AbstractVector{T}; sizepct::Integer = 100, truncate = false) where {S<:AbstractString, T<:Integer}

		feedbacklist = [digits(a, base=3, pad = 5) for a in feedbackints]
		
		winind = findfirst(==(winfeedback), feedbacklist)
	
		stopind = isnothing(winind) ? lastindex(feedbacklist) : winind
	
		gamewin = !isempty(feedbacklist) && (feedbacklist[stopind] == winfeedback)

		num_rows = truncate ? length(guesses) : 6
	
		boxclass(r, c) = "inputbox row$(r-1) box$(c-1)"
		
		extraclasses(classlist...) = isempty(classlist) ? "" : reduce((a, b) -> "$a $b", classlist)
		
		makebox(r, c, content, classlist...) = """<div class = "$(boxclass(r, c)) $(extraclasses(classlist...))" label = $(lowercase(content))></div>"""

		makefeedbackbox(r, c) = makebox(r, c, guesses[r][c], "feedback$(feedbacklist[r][c])")
		
		makewinbox(r, c) = makebox(r, c, guesses[r][c], "win")
		
		makeblankbox(r, c) = makebox(r, c, "")

		
		feedbackboxes = mapreduce(a -> makefeedbackbox(a...), add_elements, ((r, c) for r in 1:stopind-1 for c in 1:5); init = """""")
	
		winboxes = if !isempty(guesses)
			gamewin ? mapreduce(c -> makewinbox(stopind, c), add_elements, 1:5) : mapreduce(c -> makefeedbackbox(stopind, c), add_elements, 1:5)
		else
			""""""
		end
		blankboxes = mapreduce(a -> makeblankbox(a...), add_elements, ((r, c) for r in stopind+1:num_rows for c in 1:5); init= """""")
		
		HTML("""
		<div class = "wordle-game-display">
		<div class = "wordle-game-grid sizepct$sizepct">
			$feedbackboxes
			$winboxes
			$blankboxes
		</div>
		</div>
		<style>
	
			.wordle-game-grid.sizepct$sizepct {
				width: calc($(sizepct/100)*var(--container-width));
				height: calc($(sizepct/100)*var(--container-width)*$num_rows/5);
				margin: calc($(sizepct/100)*var(--container-width)/35);
				grid-gap: calc($(sizepct/100)*var(--container-width)/70);
			}
	
			.wordle-game-grid.sizepct$sizepct .inputbox {
				font-size: calc($(sizepct/100)*var(--container-width)/5/2.0);
			}
			
		
			.wordle-game-display .wordle-game-grid:hover * {
				content: '';
				animation: showstatus 0s;
			}
	
			.wordle-game-grid:hover.gamewon *, .wordle-game-grid:hover.gamelost *, .wordle-game-grid:hover.gameover * {
				content: '';
				animation: fadein 1s;
			}
		
		
			.wordle-game-grid:hover.gamewon::after {
				content: 'Congratulations!';
				color: forestgreen;
				
			}
	
			@keyframes showstatus {
				0% {opacity: 0;}
				50% {transform: scale(0.8);}
			}
		</style>
		""")
	end
end

# ╔═╡ 67d4a8f0-f96e-45f1-a8c0-25a31707551c
show_wordle_game(WordleState(["trace"], [242]); sizepct = 10)

# ╔═╡ 8128981a-2536-4958-a934-625ddc535090
show_wordle_game([example_guess], [example_guess_feedback_bytes]; truncate = true)

# ╔═╡ d2bf55fa-ac04-45d6-9d6a-8c7ac855c3a1
function visualize_afterstate_transition(s::WordleState, guess; sizepct = 100)
	w = WordleAfterState(s, guess)
	d = wordle_afterstate_transition(w)
	new_states = [a[2] for a in keys(d)]
	probabilities = [d[a] for a in keys(d)]
	HTML("""
	<div style = "font-size: 1.5em;"><b>Root State</b></div>
	$(show_wordle_game(s; truncate = true).content)
	<div style = "font-size: 1.5em;"><b>$(length(new_states)) Transition States (size proportional to probability)</b></div>
	<div style = "display:flex; flex-wrap: wrap; ">
		$(mapreduce(a -> show_wordle_game(WordleState(a[1].guess_list[end:end], a[1].feedback_list[end:end]); truncate = true, sizepct = round(Int64, a[2]*sizepct)).content, add_elements, zip(new_states, probabilities)))
	</div>
	""")
end

# ╔═╡ 52faa94f-e7ca-4e28-9ed2-0a788db5e231
visualize_afterstate_transition(isempty(example_state.guesses) ? WordleState() : WordleState(example_state.guesses, convert_bytes.(example_state.feedback)), example_new_guess; sizepct = size_adj)

# ╔═╡ Cell order:
# ╟─7be1e611-0cba-4bf8-876d-757dd3931016
# ╠═aa5ce283-909c-4752-8e1b-71b09720ae1b
# ╠═f9dc6d6f-e417-482c-842b-c3d8ddaedd0e
# ╟─5d564193-c55b-4584-96ff-5b1cf404e334
# ╟─b04c7371-c26a-4400-8dae-06b922b27af1
# ╟─589fac35-d61b-4ece-a316-610b91f26640
# ╟─959f4088-9a24-4104-ad2e-1d1a8edfc3b2
# ╠═67d4a8f0-f96e-45f1-a8c0-25a31707551c
# ╟─e460fde6-f65a-4af2-9517-327a187b112a
# ╠═8a090c51-aa61-4090-96a1-8f4833bb9983
# ╠═502552da-745c-4991-a133-6f786191b255
# ╟─cf9cf9d8-b194-4b1d-afcd-22229ab0891b
# ╠═6851d75d-075c-47f8-8da6-f207e1382ccf
# ╟─d4a07d83-cab3-4ca3-bed0-3880ade5ff6a
# ╠═05b55c5f-0471-4032-9e7a-b61c58b33ce1
# ╟─6d771e10-781c-49bd-bf66-417a751713eb
# ╟─fa02baf4-dc36-461a-a7ea-0ee22fb6011f
# ╟─451219d2-c0c6-4a28-ad27-9459cf860aa1
# ╟─8128981a-2536-4958-a934-625ddc535090
# ╟─56f7a7a1-721b-44e8-8cbb-253de9ab3d3d
# ╠═b657a428-702a-47a3-9caa-7ed4ac09e7ca
# ╠═c43af2b6-733b-4b1b-a249-bb261f059783
# ╠═6047122e-99bd-4719-b9fe-0253fe610780
# ╠═026063c3-705a-4d58-b3f6-0563485afe32
# ╠═f38a4cf6-9daf-48ad-83f4-36e95020d13f
# ╠═4541253d-09fa-4485-80e0-0d64207be03d
# ╠═d7544672-091a-4438-9b08-4d49b781dccb
# ╟─75f63d9b-6d79-4113-8ad7-cb3833d02c23
# ╟─2b3ba2f2-bc89-4226-b60e-c05306f5886b
# ╟─54347f08-3bc8-474d-9a1c-1888c4c126ea
# ╠═bb513fc5-1bdc-4c27-a778-86aa71833d0e
# ╟─86616282-8287-4371-95bb-111940c16b0f
# ╟─65995e32-cdde-49d2-a916-00a48a46ecb5
# ╠═12d7468b-2675-476a-88ce-284e85b4a589
# ╠═d3d33c45-434b-44ff-b4d7-7ba6e1f415fd
# ╟─a3121433-4d2e-4224-905b-aa0f8d91db05
# ╠═6ad205c6-7a67-4aa2-82a7-83358589ddca
# ╟─82259951-eae3-4f20-89ea-b84144115028
# ╟─8353133e-7559-48fe-b1b5-51b330e70182
# ╠═e88225c3-931f-4672-8a38-ab58116a2b75
# ╠═46c0ce87-a94f-4b15-900f-be92775ee066
# ╠═f8d41dde-4b0d-432b-8223-d55bcce94736
# ╠═60c47b36-207d-4812-bcb8-4ecb932878ab
# ╠═a95fbf96-6996-4529-96cb-4761894f4c23
# ╠═e5e62c7e-d39b-47bf-b3c3-12b898816f51
# ╠═4a8134aa-e0db-43ad-837d-c38a235b4712
# ╠═8bb972da-20e5-4d0c-b964-b56ba62e631e
# ╠═2b7adbb7-5c64-42fe-8178-3d1187f4b3fb
# ╠═3553570d-970e-4e1d-929b-19387e79a31e
# ╠═464a4be1-1fa2-4d60-9fb8-72fc47723cf5
# ╟─e5b1d8e5-f224-44e3-8190-b8146ed3ea92
# ╟─32c30d8e-de5b-43a0-bfdb-8fb86037de7f
# ╠═c1858879-2a20-4af6-af4c-a03d26dda7a3
# ╟─be061e94-d403-453a-99fb-7a1e13bebf52
# ╠═b75bb4fe-6b09-4d7c-8bc0-6bf224d2c95a
# ╠═783d068c-77b0-43e1-907b-e532317c5afd
# ╠═bb5ef5ff-93fe-4985-a920-442862e4498b
# ╟─b9a5d547-79b4-4c7c-a3eb-ed7e87513a88
# ╠═1d5ba870-3110-4576-a116-a8d0a4d84edc
# ╠═93b857e5-72db-4aaf-abb0-295beab4073c
# ╟─d2c1294e-1885-489c-98de-29930e9851c0
# ╠═d7378cfd-140f-4161-a13f-398ffbc642b7
# ╠═633e747b-3f5f-40e9-886a-590cb19d40dd
# ╠═d4f180c3-97f7-4e97-a904-a51f29207070
# ╠═852dff48-8c1d-42f2-925e-53f889b1ac6a
# ╠═4eb18ec4-b327-4f97-a380-10469441cff8
# ╠═20a41989-6650-41dd-b34e-fa23c993e669
# ╠═d9c4ff04-12ba-4e1c-bc52-4c12388d514b
# ╠═97d40991-d719-4e87-bd70-94551a645448
# ╟─92cb4e96-714d-425d-a64b-eff26a5f92ef
# ╠═a383cd80-dd49-4f9f-ba67-1f58ea7eb0b6
# ╠═2cda2a2c-caf2-474c-90a4-388f15260501
# ╟─f2ab3716-19a2-4cbb-b46f-a13c297e086d
# ╠═86f6910f-0187-4d38-afff-295ec67fd0c0
# ╠═35025fe1-e98b-46c6-b159-37d255100b73
# ╠═b1e3f7ea-e2de-4467-b301-0c3cf225e433
# ╠═bd744b6d-db57-42bf-88c0-c667dfb03f5f
# ╟─8e2477a9-ca14-4ecf-9194-9bac5ef25fd4
# ╟─70dcac79-2839-4c1d-8f48-cbe294b3ffb0
# ╟─bf1c13cd-a066-4865-b43f-0cba3d3df8d2
# ╟─69aacb14-52b2-49a2-a2bc-e8c07b9e1986
# ╠═52faa94f-e7ca-4e28-9ed2-0a788db5e231
# ╠═d2bf55fa-ac04-45d6-9d6a-8c7ac855c3a1
# ╠═961593ef-964c-4363-b4ba-0d45cfe3d198
# ╠═a3b9f0c9-c8bd-4237-8b1c-a1fed136e988
# ╠═d3d0df67-ec26-4352-9164-a53d14d1b065
# ╠═f6253866-7bef-400f-9713-e0fd5054b201
# ╠═cda5622b-f4f7-4ec2-a848-ff5eb9c9b66b
# ╠═9a75b05d-82fd-4a3c-8cb9-5161dfc18949
# ╠═73d4533a-1d8a-4b3a-a4ed-a9f64cc5ee23
# ╠═05ffe4f9-8cd6-424e-8955-e3ca21c486f3
# ╠═b0e2ea69-13a9-4520-920b-bf5089cc8da8
# ╠═28709a4e-67cd-4dff-aad3-cb8af93038d7
# ╠═379e4103-ca2a-4711-9df7-d6cd83afe653
# ╠═8cb865b4-8928-4210-96d5-6d6a71deb03b
# ╠═09c89ae6-83e2-4f86-8c3f-b1528235c70a
# ╠═2fa84583-4ff7-48d0-bd62-cb7380535e91
# ╟─83302fc6-49ff-4d25-9fd7-0b491b79fc73
# ╟─01be89b2-6233-4e11-b142-fa6ad05880a3
# ╟─4578c627-127e-4a18-972c-03b38cff371b
# ╟─57281f5c-4c93-4308-b1e7-e5a890655262
# ╠═52e5d54d-c40d-48e1-943f-01985e9c4f7f
# ╟─9f9b8db2-7c74-49f0-b067-bb6aa433fbe0
# ╟─faade165-8066-4deb-93a4-2b7daabd57dc
# ╠═e804f5fc-c817-4851-a14b-fdcca1180773
# ╟─7f1f2057-24ca-44c0-8496-b6ef68d895bd
# ╠═9ba4f6e0-f0cc-4b8f-b70e-8e9a06aa03ad
# ╠═e093dd57-040b-4983-a95f-057097fcefff
# ╠═06138a4e-67a3-47a3-84a5-92013fd404ca
# ╠═889a2d4b-1adf-44cd-8d3f-22d4aa4c3de3
# ╠═87e696f6-a9f5-4e00-8678-ec81ad530dd0
# ╠═10d1f403-34c8-46ce-8cfc-d289608f465c
# ╟─caaeeaee-94b7-498d-a746-a2c5c7177347
# ╟─fb8439a3-ee95-487f-a755-ffcbd8b3c381
# ╟─d9d54e56-a32f-4ffb-820d-df0b7918c78c
# ╟─7cd9cb06-4731-4eaa-b745-32118278d360
# ╟─df9a4dd8-5a36-497b-a61f-d76a830f4398
# ╠═41e0cddb-e78d-477b-849a-124754340a3c
# ╟─03bbb910-dfa7-4c28-b811-afa9e5ca0e63
# ╟─cb3b46cf-8375-43b2-8736-97882e9b5e18
# ╠═6fdc99b1-beea-4c45-98bb-d257501a6878
# ╟─aaf516b5-f982-44c3-bcae-14d46ad72e82
# ╠═2e75e7a8-66e7-4228-a85f-6b32ba933018
# ╠═4225d99f-c30f-47e8-b6c1-9a167d4e937c
# ╟─ee646ead-3600-49ed-b7ed-3c9a8af7d195
# ╟─58316d89-7149-42fe-802f-ecd44999ea77
# ╠═731c6a58-b223-48f3-82dc-b7cf3772c699
# ╠═f79720db-a5dc-4267-a209-181e15e6f200
# ╠═028ad58f-f34b-4741-a99e-42c86f88bd74
# ╠═15bc3ba3-8fa7-4c62-91b6-71e797e4a83d
# ╠═286b4ce9-5412-4506-8334-983655e3ecba
# ╠═a78a462c-647a-477d-abcc-aaeae1202cca
# ╠═9c898c04-65a3-44b0-bbfa-bfcf4ca32c0f
# ╠═9cf9cb1c-62af-4dde-b28b-d6733c1432b0
# ╠═7c968494-0e3e-47e4-9666-6d1f7e5a9e85
# ╠═37209bdb-8f5c-4e4b-8e88-c77a6a07fdfd
# ╠═8d9cb914-1f6b-4ed0-b84b-783727a8c3e8
# ╠═501e5240-8d3d-4db4-b053-1c4148cd128f
# ╠═80ab2d58-787a-4916-b2d1-2d56d8f212d9
# ╠═d7c69c40-5ae5-4f95-bff6-c247cb2d1c33
# ╠═704d5d28-3ded-4d1a-a89e-4e636b88cd05
# ╠═b3bd847c-f21f-425c-ae05-abd95da6e858
# ╠═7804e77c-f525-4ecd-a91e-fbd9d6ccf22a
# ╠═0e73c1ad-89b1-40d3-842e-398c98ecce14
# ╟─64014b10-9974-475a-8625-fef445fe8e4f
# ╠═e426b806-4435-49b9-824a-04fe34a48e9e
# ╠═10eb71ed-df57-49b2-ae98-336191346b29
# ╠═b9beca61-7efc-4186-81c2-1ceda103c801
# ╠═bab1c451-bf1b-495c-9448-0be284063733
# ╠═ccca515b-17ad-4469-ac57-61c2f3f86e62
# ╠═67e57ae3-9834-43e6-ad46-91a6c046b69b
# ╠═1baafa2c-b087-41be-9e5f-6adea8a985fc
# ╟─d7b0a4ba-ba18-41ad-ade3-dde119f08a13
# ╠═7e8be36e-5237-49f6-95ec-b9cb24b32c34

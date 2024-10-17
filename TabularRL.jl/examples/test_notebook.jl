### A Pluto.jl notebook ###
# v0.19.46

using Markdown
using InteractiveUtils

# ╔═╡ 16f8f288-9371-44ae-914d-4ea13fec98f6
using PlutoDevMacros

# ╔═╡ 2deb31ad-ab39-4f27-885e-79bfccce97e3
PlutoDevMacros.@fromparent begin
	using TabularRL
	using >.SparseArrays, >.Random, >.Statistics, >.StaticArrays, >.Transducers, >.Serialization
end

# ╔═╡ e247734e-d1e4-43f2-a74e-0d0bd5971a4b
using LinearAlgebra, StatsBase, Base.Threads, DataStructures, DataFrames, JLD2

# ╔═╡ afbace21-a366-4ef1-9b50-198381293d22
# ╠═╡ skip_as_script = true
#=╠═╡
begin
	using Markdown, PlutoHooks, HypertextLiteral, PlutoUI, PlutoPlotly, PlutoProfile, BenchmarkTools
	import AbstractPlutoDingetjes.Bonds
	TableOfContents()
end
  ╠═╡ =#

# ╔═╡ 5d564193-c55b-4584-96ff-5b1cf404e334
#=╠═╡
md"""
# Test Environments
"""
  ╠═╡ =#

# ╔═╡ b04c7371-c26a-4400-8dae-06b922b27af1
#=╠═╡
md"""
## Wordle Environment
"""
  ╠═╡ =#

# ╔═╡ 589fac35-d61b-4ece-a316-610b91f26640
#=╠═╡
md"""
### Game Description

Wordle is a game of guessing 5 letter words from a predefined pool of allowed guesses.  A player has six attempts to guess the correct word and the game ends after a correct guess or when six incorrect guesses have been made.  After each guess, the player receives feedback per letter according to the following rules:

- Green: A letter is colored green if it matches the position in the answer
- Yellow: A letter is colored yellow if it is present in the answer but in a different location.  If the letter appears once in the answer but multiple times in the guess, only the first instance of the letter will be yellow.  If a letter appears more than once in the answer, then it could appear yellow multiple times in the guess as well.  This feedback is the most nuanced given its behavior changes depending on previous letters including if any were marked green
- Gray: A letter is marked gray if it does not appear in the answer

Throughout the game, a player can see the feedback for each prior guess.  A correct guess will receive the unique feedback of five green letters.
"""
  ╠═╡ =#

# ╔═╡ 959f4088-9a24-4104-ad2e-1d1a8edfc3b2
#=╠═╡
md"""
### Letters and Feedback

Before creating the MDP, it is important to save precomputed values and structures to make step evaluations easier
"""
  ╠═╡ =#

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
#=╠═╡
md"""
### Example Feedback and Display
"""
  ╠═╡ =#

# ╔═╡ 6851d75d-075c-47f8-8da6-f207e1382ccf
const example_feedback = [MISSING, EXACT, MISPLACED, MISPLACED, MISSING]

# ╔═╡ 6d771e10-781c-49bd-bf66-417a751713eb
#=╠═╡
md"""
### Wordle Feedback Visualization Example
"""
  ╠═╡ =#

# ╔═╡ 451219d2-c0c6-4a28-ad27-9459cf860aa1
#=╠═╡
md"""
#### Observed feedback in game
"""
  ╠═╡ =#

# ╔═╡ 56f7a7a1-721b-44e8-8cbb-253de9ab3d3d
#=╠═╡
md"""
#### Numerical feedback values
"""
  ╠═╡ =#

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
#=╠═╡
md"""
###  Word Data
"""
  ╠═╡ =#

# ╔═╡ 2b3ba2f2-bc89-4226-b60e-c05306f5886b
#=╠═╡
md"""
The following words were the original answers for the game which were predefined for each day.  Since the game was acquired by the New York Times, there is no longer a predefined answer list.
"""
  ╠═╡ =#

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
#=╠═╡
md"""
The following words are allowed guesses for the New York Times Wordle game retrieved on March 27, 2023
"""
  ╠═╡ =#

# ╔═╡ 65995e32-cdde-49d2-a916-00a48a46ecb5
#allowed guesses embedded in the NYT wordle source code as of 05/27/2023
const nyt_valid_words = ["aahed","aalii","aapas","aargh","aarti","abaca","abaci","abacs","abaft","abaht","abaka","abamp","aband","abash","abask","abaya","abbas","abbed","abbes","abcee","abeam","abear","abeat","abeer","abele","abeng","abers","abets","abeys","abies","abius","abjad","abjud","abler","ables","ablet","ablow","abmho","abnet","abohm","aboil","aboma","aboon","abord","abore","aborn","abram","abray","abrim","abrin","abris","absey","absit","abuna","abune","abura","aburn","abuts","abuzz","abyes","abysm","acais","acara","acari","accas","accha","accoy","accra","acedy","acene","acerb","acers","aceta","achar","ached","acher","aches","achey","achoo","acids","acidy","acies","acing","acini","ackee","acker","acmes","acmic","acned","acnes","acock","acoel","acold","acone","acral","acred","acres","acron","acros","acryl","actas","acted","actin","acton","actus","acyls","adats","adawn","adaws","adays","adbot","addas","addax","added","adder","addin","addio","addle","addra","adead","adeem","adhan","adhoc","adieu","adios","adits","adlib","adman","admen","admix","adnex","adobo","adoon","adorb","adown","adoze","adrad","adraw","adred","adret","adrip","adsum","aduki","adunc","adust","advew","advts","adyta","adyts","adzed","adzes","aecia","aedes","aeger","aegis","aeons","aerie","aeros","aesir","aevum","afald","afanc","afara","afars","afear","affly","afion","afizz","aflaj","aflap","aflow","afoam","afore","afret","afrit","afros","aftos","agals","agama","agami","agamy","agars","agasp","agast","agaty","agave","agaze","agbas","agene","agers","aggag","agger","aggie","aggri","aggro","aggry","aghas","agidi","agila","agios","agism","agist","agita","aglee","aglet","agley","agloo","aglus","agmas","agoge","agogo","agone","agons","agood","agora","agria","agrin","agros","agrum","agued","agues","aguey","aguna","agush","aguti","aheap","ahent","ahigh","ahind","ahing","ahint","ahold","ahole","ahull","ahuru","aidas","aided","aides","aidoi","aidos","aiery","aigas","aight","ailed","aimag","aimak","aimed","aimer","ainee","ainga","aioli","aired","airer","airns","airth","airts","aitch","aitus","aiver","aixes","aiyah","aiyee","aiyoh","aiyoo","aizle","ajies","ajiva","ajuga","ajupa","ajwan","akara","akees","akela","akene","aking","akita","akkas","akker","akoia","akoja","akoya","aksed","akses","alaap","alack","alala","alamo","aland","alane","alang","alans","alant","alapa","alaps","alary","alata","alate","alays","albas","albee","albid","alcea","alces","alcid","alcos","aldea","alder","aldol","aleak","aleck","alecs","aleem","alefs","aleft","aleph","alews","aleye","alfas","algal","algas","algid","algin","algor","algos","algum","alias","alick","alifs","alims","aline","alios","alist","aliya","alkie","alkin","alkos","alkyd","alkyl","allan","allee","allel","allen","aller","allin","allis","allod","allus","allyl","almah","almas","almeh","almes","almud","almug","alods","aloed","aloes","aloha","aloin","aloos","alose","alowe","altho","altos","alula","alums","alumy","alure","alurk","alvar","alway","amahs","amain","amari","amaro","amate","amaut","amban","ambit","ambos","ambry","ameba","ameer","amene","amens","ament","amias","amice","amici","amide","amido","amids","amies","amiga","amigo","amine","amino","amins","amirs","amlas","amman","ammas","ammon","ammos","amnia","amnic","amnio","amoks","amole","amore","amort","amour","amove","amowt","amped","ampul","amrit","amuck","amyls","anana","anata","ancho","ancle","ancon","andic","andro","anear","anele","anent","angas","anglo","anigh","anile","anils","anima","animi","anion","anise","anker","ankhs","ankus","anlas","annal","annan","annas","annat","annum","annus","anoas","anole","anomy","ansae","ansas","antae","antar","antas","anted","antes","antis","antra","antre","antsy","anura","anyon","apace","apage","apaid","apayd","apays","apeak","apeek","apers","apert","apery","apgar","aphis","apian","apiol","apish","apism","apode","apods","apols","apoop","aport","appal","appam","appay","appel","appro","appts","appui","appuy","apres","apses","apsis","apsos","apted","apter","aquae","aquas","araba","araks","arame","arars","arbah","arbas","arced","archi","arcos","arcus","ardeb","ardri","aread","areae","areal","arear","areas","areca","aredd","arede","arefy","areic","arene","arepa","arere","arete","arets","arett","argal","argan","argil","argle","argol","argon","argot","argus","arhat","arias","ariel","ariki","arils","ariot","arish","arith","arked","arled","arles","armed","armer","armet","armil","arnas","arnis","arnut","aroba","aroha","aroid","arpas","arpen","arrah","arras","arret","arris","arroz","arsed","arses","arsey","arsis","artal","artel","arter","artic","artis","artly","aruhe","arums","arval","arvee","arvos","aryls","asada","asana","ascon","ascus","asdic","ashed","ashes","ashet","asity","askar","asked","asker","askoi","askos","aspen","asper","aspic","aspie","aspis","aspro","assai","assam","assed","asses","assez","assot","aster","astir","astun","asura","asway","aswim","asyla","ataps","ataxy","atigi","atilt","atimy","atlas","atman","atmas","atmos","atocs","atoke","atoks","atoms","atomy","atony","atopy","atria","atrip","attap","attar","attas","atter","atuas","aucht","audad","audax","augen","auger","auges","aught","aulas","aulic","auloi","aulos","aumil","aunes","aunts","aurae","aural","aurar","auras","aurei","aures","auric","auris","aurum","autos","auxin","avale","avant","avast","avels","avens","avers","avgas","avine","avion","avise","aviso","avize","avows","avyze","awari","awarn","awato","awave","aways","awdls","aweel","aweto","awing","awkin","awmry","awned","awner","awols","awork","axels","axile","axils","axing","axite","axled","axles","axman","axmen","axoid","axone","axons","ayahs","ayaya","ayelp","aygre","ayins","aymag","ayont","ayres","ayrie","azans","azide","azido","azine","azlon","azoic","azole","azons","azote","azoth","azuki","azurn","azury","azygy","azyme","azyms","baaed","baals","baaps","babas","babby","babel","babes","babka","baboo","babul","babus","bacca","bacco","baccy","bacha","bachs","backs","backy","bacne","badam","baddy","baels","baffs","baffy","bafta","bafts","baghs","bagie","bagsy","bagua","bahts","bahus","bahut","baiks","baile","bails","bairn","baisa","baith","baits","baiza","baize","bajan","bajra","bajri","bajus","baked","baken","bakes","bakra","balas","balds","baldy","baled","bales","balks","balky","ballo","balls","bally","balms","baloi","balon","baloo","balot","balsa","balti","balun","balus","balut","bamas","bambi","bamma","bammy","banak","banco","bancs","banda","bandh","bands","bandy","baned","banes","bangs","bania","banks","banky","banns","bants","bantu","banty","bantz","banya","baons","baozi","bappu","bapus","barbe","barbs","barby","barca","barde","bardo","bards","bardy","bared","barer","bares","barfi","barfs","barfy","baric","barks","barky","barms","barmy","barns","barny","barps","barra","barre","barro","barry","barye","basan","basas","based","basen","baser","bases","basha","basho","basij","basks","bason","basse","bassi","basso","bassy","basta","basti","basto","basts","bated","bates","baths","batik","batos","batta","batts","battu","bauds","bauks","baulk","baurs","bavin","bawds","bawks","bawls","bawns","bawrs","bawty","bayas","bayed","bayer","bayes","bayle","bayts","bazar","bazas","bazoo","bball","bdays","beads","beaks","beaky","beals","beams","beamy","beano","beans","beany","beare","bears","beath","beats","beaty","beaus","beaut","beaux","bebop","becap","becke","becks","bedad","bedel","bedes","bedew","bedim","bedye","beedi","beefs","beeps","beers","beery","beets","befog","begad","begar","begem","begob","begot","begum","beige","beigy","beins","beira","beisa","bekah","belah","belar","belay","belee","belga","belit","belli","bello","bells","belon","belts","belve","bemad","bemas","bemix","bemud","bends","bendy","benes","benet","benga","benis","benji","benne","benni","benny","bento","bents","benty","bepat","beray","beres","bergs","berko","berks","berme","berms","berob","beryl","besat","besaw","besee","beses","besit","besom","besot","besti","bests","betas","beted","betes","beths","betid","beton","betta","betty","bevan","bever","bevor","bevue","bevvy","bewdy","bewet","bewig","bezes","bezil","bezzy","bhais","bhaji","bhang","bhats","bhava","bhels","bhoot","bhuna","bhuts","biach","biali","bialy","bibbs","bibes","bibis","biccy","bices","bicky","bided","bider","bides","bidet","bidis","bidon","bidri","bield","biers","biffo","biffs","biffy","bifid","bigae","biggs","biggy","bigha","bight","bigly","bigos","bihon","bijou","biked","biker","bikes","bikie","bikky","bilal","bilat","bilbo","bilby","biled","biles","bilgy","bilks","bills","bimah","bimas","bimbo","binal","bindi","binds","biner","bines","bings","bingy","binit","binks","binky","bints","biogs","bions","biont","biose","biota","biped","bipod","bippy","birdo","birds","biris","birks","birle","birls","biros","birrs","birse","birsy","birze","birzz","bises","bisks","bisom","bitch","biter","bites","bitey","bitos","bitou","bitsy","bitte","bitts","bivia","bivvy","bizes","bizzo","bizzy","blabs","blads","blady","blaer","blaes","blaff","blags","blahs","blain","blams","blanc","blart","blase","blash","blate","blats","blatt","blaud","blawn","blaws","blays","bleah","blear","blebs","blech","blees","blent","blert","blest","blets","bleys","blimy","bling","blini","blins","bliny","blips","blist","blite","blits","blive","blobs","blocs","blogs","blonx","blook","bloop","blore","blots","blows","blowy","blubs","blude","bluds","bludy","blued","blues","bluet","bluey","bluid","blume","blunk","blurs","blype","boabs","boaks","boars","boart","boats","boaty","bobac","bobak","bobas","bobol","bobos","bocca","bocce","bocci","boche","bocks","boded","bodes","bodge","bodgy","bodhi","bodle","bodoh","boeps","boers","boeti","boets","boeuf","boffo","boffs","bogan","bogey","boggy","bogie","bogle","bogue","bogus","bohea","bohos","boils","boing","boink","boite","boked","bokeh","bokes","bokos","bolar","bolas","boldo","bolds","boles","bolet","bolix","bolks","bolls","bolos","bolts","bolus","bomas","bombe","bombo","bombs","bomoh","bomor","bonce","bonds","boned","boner","bones","bongs","bonie","bonks","bonne","bonny","bonum","bonza","bonze","booai","booay","boobs","boody","booed","boofy","boogy","boohs","books","booky","bools","booms","boomy","boong","boons","boord","boors","boose","boots","boppy","borak","boral","boras","borde","bords","bored","boree","borek","borel","borer","bores","borgo","boric","borks","borms","borna","boron","borts","borty","bortz","bosey","bosie","bosks","bosky","boson","bossa","bosun","botas","boteh","botel","botes","botew","bothy","botos","botte","botts","botty","bouge","bouks","boult","bouns","bourd","bourg","bourn","bouse","bousy","bouts","boutu","bovid","bowat","bowed","bower","bowes","bowet","bowie","bowls","bowne","bowrs","bowse","boxed","boxen","boxes","boxla","boxty","boyar","boyau","boyed","boyey","boyfs","boygs","boyla","boyly","boyos","boysy","bozos","braai","brach","brack","bract","brads","braes","brags","brahs","brail","braks","braky","brame","brane","brank","brans","brant","brast","brats","brava","bravi","braws","braxy","brays","braza","braze","bream","brede","breds","breem","breer","brees","breid","breis","breme","brens","brent","brere","brers","breve","brews","breys","brier","bries","brigs","briki","briks","brill","brims","brins","brios","brise","briss","brith","brits","britt","brize","broch","brock","brods","brogh","brogs","brome","bromo","bronc","brond","brool","broos","brose","brosy","brows","bruck","brugh","bruhs","bruin","bruit","bruja","brujo","brule","brume","brung","brusk","brust","bruts","bruvs","buats","buaze","bubal","bubas","bubba","bubbe","bubby","bubus","buchu","bucko","bucks","bucku","budas","buded","budes","budis","budos","buena","buffa","buffe","buffi","buffo","buffs","buffy","bufos","bufty","bugan","buhls","buhrs","buiks","buist","bukes","bukos","bulbs","bulgy","bulks","bulla","bulls","bulse","bumbo","bumfs","bumph","bumps","bumpy","bunas","bunce","bunco","bunde","bundh","bunds","bundt","bundu","bundy","bungs","bungy","bunia","bunje","bunjy","bunko","bunks","bunns","bunts","bunty","bunya","buoys","buppy","buran","buras","burbs","burds","buret","burfi","burgh","burgs","burin","burka","burke","burks","burls","burns","buroo","burps","burqa","burra","burro","burrs","burry","bursa","burse","busby","buses","busks","busky","bussu","busti","busts","busty","buteo","butes","butle","butoh","butts","butty","butut","butyl","buyin","buzzy","bwana","bwazi","byded","bydes","byked","bykes","byres","byrls","byssi","bytes","byway","caaed","cabas","caber","cabob","caboc","cabre","cacas","cacks","cacky","cadee","cades","cadge","cadgy","cadie","cadis","cadre","caeca","caese","cafes","caffe","caffs","caged","cager","cages","cagot","cahow","caids","cains","caird","cajon","cajun","caked","cakes","cakey","calfs","calid","calif","calix","calks","calla","calle","calls","calms","calmy","calos","calpa","calps","calve","calyx","caman","camas","cames","camis","camos","campi","campo","camps","campy","camus","cando","caned","caneh","caner","canes","cangs","canid","canna","canns","canso","canst","canti","canto","cants","canty","capas","capax","caped","capes","capex","caphs","capiz","caple","capon","capos","capot","capri","capul","carap","carbo","carbs","carby","cardi","cards","cardy","cared","carer","cares","caret","carex","carks","carle","carls","carne","carns","carny","carob","carom","caron","carpe","carpi","carps","carrs","carse","carta","carte","carts","carvy","casas","casco","cased","caser","cases","casks","casky","casts","casus","cates","cauda","cauks","cauld","cauls","caums","caups","cauri","causa","cavas","caved","cavel","caver","caves","cavie","cavus","cawed","cawks","caxon","ceaze","cebid","cecal","cecum","ceded","ceder","cedes","cedis","ceiba","ceili","ceils","celeb","cella","celli","cells","celly","celom","celts","cense","cento","cents","centu","ceorl","cepes","cerci","cered","ceres","cerge","ceria","ceric","cerne","ceroc","ceros","certs","certy","cesse","cesta","cesti","cetes","cetyl","cezve","chaap","chaat","chace","chack","chaco","chado","chads","chaft","chais","chals","chams","chana","chang","chank","chape","chaps","chapt","chara","chare","chark","charr","chars","chary","chats","chava","chave","chavs","chawk","chawl","chaws","chaya","chays","cheba","chedi","cheeb","cheep","cheet","chefs","cheka","chela","chelp","chemo","chems","chere","chert","cheth","chevy","chews","chewy","chiao","chias","chiba","chibs","chica","chich","chico","chics","chiel","chiko","chiks","chile","chimb","chimo","chimp","chine","ching","chink","chino","chins","chips","chirk","chirl","chirm","chiro","chirr","chirt","chiru","chiti","chits","chiva","chive","chivs","chivy","chizz","choco","chocs","chode","chogs","choil","choko","choky","chola","choli","cholo","chomp","chons","choof","chook","choom","choon","chops","choss","chota","chott","chout","choux","chowk","chows","chubs","chufa","chuff","chugs","chums","churl","churr","chuse","chuts","chyle","chyme","chynd","cibol","cided","cides","ciels","ciggy","cilia","cills","cimar","cimex","cinct","cines","cinqs","cions","cippi","circs","cires","cirls","cirri","cisco","cissy","cists","cital","cited","citee","citer","cites","cives","civet","civie","civvy","clach","clade","clads","claes","clags","clair","clame","clams","clans","claps","clapt","claro","clart","clary","clast","clats","claut","clave","clavi","claws","clays","cleck","cleek","cleep","clefs","clegs","cleik","clems","clepe","clept","cleve","clews","clied","clies","clift","clime","cline","clint","clipe","clips","clipt","clits","cloam","clods","cloff","clogs","cloke","clomb","clomp","clonk","clons","cloop","cloot","clops","clote","clots","clour","clous","clows","cloye","cloys","cloze","clubs","clues","cluey","clunk","clype","cnida","coact","coady","coala","coals","coaly","coapt","coarb","coate","coati","coats","cobbs","cobby","cobia","coble","cobot","cobza","cocas","cocci","cocco","cocks","cocky","cocos","cocus","codas","codec","coded","coden","coder","codes","codex","codon","coeds","coffs","cogie","cogon","cogue","cohab","cohen","cohoe","cohog","cohos","coifs","coign","coils","coins","coirs","coits","coked","cokes","cokey","colas","colby","colds","coled","coles","coley","colic","colin","colle","colls","colly","colog","colts","colza","comae","comal","comas","combe","combi","combo","combs","comby","comer","comes","comix","comme","commo","comms","commy","compo","comps","compt","comte","comus","coned","cones","conex","coney","confs","conga","conge","congo","conia","conin","conks","conky","conne","conns","conte","conto","conus","convo","cooch","cooed","cooee","cooer","cooey","coofs","cooks","cooky","cools","cooly","coomb","cooms","coomy","coons","coops","coopt","coost","coots","cooty","cooze","copal","copay","coped","copen","coper","copes","copha","coppy","copra","copsy","coqui","coram","corbe","corby","corda","cords","cored","cores","corey","corgi","coria","corks","corky","corms","corni","corno","corns","cornu","corps","corse","corso","cosec","cosed","coses","coset","cosey","cosie","costa","coste","costs","cotan","cotch","coted","cotes","coths","cotta","cotts","coude","coups","courb","courd","coure","cours","couta","couth","coved","coves","covin","cowal","cowan","cowed","cowks","cowls","cowps","cowry","coxae","coxal","coxed","coxes","coxib","coyau","coyed","coyer","coypu","cozed","cozen","cozes","cozey","cozie","craal","crabs","crags","craic","craig","crake","crame","crams","crans","crape","craps","crapy","crare","craws","crays","creds","creel","crees","crein","crema","crems","crena","creps","crepy","crewe","crews","crias","cribo","cribs","cries","crims","crine","crink","crins","crios","cripe","crips","crise","criss","crith","crits","croci","crocs","croft","crogs","cromb","crome","cronk","crons","crool","croon","crops","crore","crost","crout","crowl","crows","croze","cruck","crudo","cruds","crudy","crues","cruet","cruft","crunk","cruor","crura","cruse","crusy","cruve","crwth","cryer","cryne","ctene","cubby","cubeb","cubed","cuber","cubes","cubit","cucks","cudda","cuddy","cueca","cuffo","cuffs","cuifs","cuing","cuish","cuits","cukes","culch","culet","culex","culls","cully","culms","culpa","culti","cults","culty","cumec","cundy","cunei","cunit","cunny","cunts","cupel","cupid","cuppa","cuppy","cupro","curat","curbs","curch","curds","curdy","cured","curer","cures","curet","curfs","curia","curie","curli","curls","curns","curny","currs","cursi","curst","cusec","cushy","cusks","cusps","cuspy","cusso","cusum","cutch","cuter","cutes","cutey","cutin","cutis","cutto","cutty","cutup","cuvee","cuzes","cwtch","cyano","cyans","cycad","cycas","cyclo","cyder","cylix","cymae","cymar","cymas","cymes","cymol","cysts","cytes","cyton","czars","daals","dabba","daces","dacha","dacks","dadah","dadas","dadis","dadla","dados","daffs","daffy","dagga","daggy","dagos","dahis","dahls","daiko","daine","daint","daker","daled","dalek","dales","dalis","dalle","dalts","daman","damar","dames","damme","damna","damns","damps","dampy","dancy","danda","dangs","danio","danks","danny","danse","dants","dappy","daraf","darbs","darcy","dared","darer","dares","darga","dargs","daric","daris","darks","darky","darls","darns","darre","darts","darzi","dashi","dashy","datal","dated","dater","dates","datil","datos","datto","daube","daubs","dauby","dauds","dault","daurs","dauts","daven","davit","dawah","dawds","dawed","dawen","dawgs","dawks","dawns","dawts","dayal","dayan","daych","daynt","dazed","dazer","dazes","dbags","deads","deair","deals","deans","deare","dearn","dears","deary","deash","deave","deaws","deawy","debag","debby","debel","debes","debts","debud","debur","debus","debye","decad","decaf","decan","decim","decko","decks","decos","decyl","dedal","deeds","deedy","deely","deems","deens","deeps","deere","deers","deets","deeve","deevs","defat","deffo","defis","defog","degas","degum","degus","deice","deids","deify","deils","deink","deism","deist","deked","dekes","dekko","deled","deles","delfs","delft","delis","della","dells","delly","delos","delph","delts","deman","demes","demic","demit","demob","demoi","demos","demot","dempt","denar","denay","dench","denes","denet","denis","dente","dents","deoch","deoxy","derat","deray","dered","deres","derig","derma","derms","derns","derny","deros","derpy","derro","derry","derth","dervs","desex","deshi","desis","desks","desse","detag","devas","devel","devis","devon","devos","devot","dewan","dewar","dewax","dewed","dexes","dexie","dexys","dhaba","dhaks","dhals","dhikr","dhobi","dhole","dholl","dhols","dhoni","dhoti","dhows","dhuti","diact","dials","diana","diane","diazo","dibbs","diced","dicer","dices","dicht","dicks","dicky","dicot","dicta","dicto","dicts","dictu","dicty","diddy","didie","didis","didos","didst","diebs","diels","diene","diets","diffs","dight","dikas","diked","diker","dikes","dikey","dildo","dilli","dills","dimbo","dimer","dimes","dimps","dinar","dined","dines","dinge","dings","dinic","dinks","dinky","dinlo","dinna","dinos","dints","dioch","diols","diota","dippy","dipso","diram","direr","dirke","dirks","dirls","dirts","disas","disci","discs","dishy","disks","disme","dital","ditas","dited","dites","ditsy","ditts","ditzy","divan","divas","dived","dives","divey","divis","divna","divos","divot","divvy","diwan","dixie","dixit","diyas","dizen","djinn","djins","doabs","doats","dobby","dobes","dobie","dobla","doble","dobra","dobro","docht","docks","docos","docus","doddy","dodos","doeks","doers","doest","doeth","doffs","dogal","dogan","doges","dogey","doggo","doggy","dogie","dogly","dohyo","doilt","doily","doits","dojos","dolce","dolci","doled","dolee","doles","doley","dolia","dolie","dolls","dolma","dolor","dolos","dolts","domal","domed","domes","domic","donah","donas","donee","doner","donga","dongs","donko","donna","donne","donny","donsy","doobs","dooce","doody","doofs","dooks","dooky","doole","dools","dooly","dooms","doomy","doona","doorn","doors","doozy","dopas","doped","doper","dopes","doppe","dorad","dorba","dorbs","doree","dores","doric","doris","dorje","dorks","dorky","dorms","dormy","dorps","dorrs","dorsa","dorse","dorts","dorty","dosai","dosas","dosed","doseh","doser","doses","dosha","dotal","doted","doter","dotes","dotty","douar","douce","doucs","douks","doula","douma","doums","doups","doura","douse","douts","doved","doven","dover","doves","dovie","dowak","dowar","dowds","dowed","dower","dowfs","dowie","dowle","dowls","dowly","downa","downs","dowps","dowse","dowts","doxed","doxes","doxie","doyen","doyly","dozed","dozer","dozes","drabs","drack","draco","draff","drags","drail","drams","drant","draps","drapy","drats","drave","draws","drays","drear","dreck","dreed","dreer","drees","dregs","dreks","drent","drere","drest","dreys","dribs","drice","dries","drily","drips","dript","drock","droid","droil","droke","drole","drome","drony","droob","droog","drook","drops","dropt","drouk","drows","drubs","drugs","drums","drupe","druse","drusy","druxy","dryad","dryas","dsobo","dsomo","duads","duals","duans","duars","dubbo","dubby","ducal","ducat","duces","ducks","ducky","ducti","ducts","duddy","duded","dudes","duels","duets","duett","duffs","dufus","duing","duits","dukas","duked","dukes","dukka","dukun","dulce","dules","dulia","dulls","dulse","dumas","dumbo","dumbs","dumka","dumky","dumps","dunam","dunch","dunes","dungs","dungy","dunks","dunno","dunny","dunsh","dunts","duomi","duomo","duped","duper","dupes","duple","duply","duppy","dural","duras","dured","dures","durgy","durns","duroc","duros","duroy","durra","durrs","durry","durst","durum","durzi","dusks","dusts","duxes","dwaal","dwale","dwalm","dwams","dwamy","dwang","dwaum","dweeb","dwile","dwine","dyads","dyers","dyked","dykes","dykey","dykon","dynel","dynes","dynos","dzhos","eagly","eagre","ealed","eales","eaned","eards","eared","earls","earns","earnt","earst","eased","easer","eases","easle","easts","eathe","eatin","eaved","eaver","eaves","ebank","ebbed","ebbet","ebena","ebene","ebike","ebons","ebook","ecads","ecard","ecash","eched","eches","echos","ecigs","ecole","ecrus","edema","edged","edger","edges","edile","edits","educe","educt","eejit","eensy","eeven","eever","eevns","effed","effer","efits","egads","egers","egest","eggar","egged","egger","egmas","ehing","eider","eidos","eigne","eiked","eikon","eilds","eiron","eisel","ejido","ekdam","ekkas","elain","eland","elans","elchi","eldin","eleet","elemi","elfed","eliad","elint","elmen","eloge","elogy","eloin","elops","elpee","elsin","elute","elvan","elven","elver","elves","emacs","embar","embay","embog","embow","embox","embus","emeer","emend","emerg","emery","emeus","emics","emirs","emits","emmas","emmer","emmet","emmew","emmys","emoji","emong","emote","emove","empts","emule","emure","emyde","emyds","enarm","enate","ended","ender","endew","endue","enews","enfix","eniac","enlit","enmew","ennog","enoki","enols","enorm","enows","enrol","ensew","ensky","entia","entre","enure","enurn","envoi","enzym","eolid","eorls","eosin","epact","epees","epena","epene","ephah","ephas","ephod","ephor","epics","epode","epopt","eppie","epris","eques","equid","erbia","erevs","ergon","ergos","ergot","erhus","erica","erick","erics","ering","erned","ernes","erose","erred","erses","eruct","erugo","eruvs","erven","ervil","escar","escot","esile","eskar","esker","esnes","esrog","esses","estoc","estop","estro","etage","etape","etats","etens","ethal","ethne","ethyl","etics","etnas","etrog","ettin","ettle","etuis","etwee","etyma","eughs","euked","eupad","euros","eusol","evegs","evens","evert","evets","evhoe","evils","evite","evohe","ewers","ewest","ewhow","ewked","exams","exeat","execs","exeem","exeme","exfil","exier","exies","exine","exing","exite","exits","exode","exome","exons","expat","expos","exude","exuls","exurb","eyass","eyers","eyots","eyras","eyres","eyrie","eyrir","ezine","fabbo","fabby","faced","facer","faces","facey","facia","facie","facta","facto","facts","facty","faddy","faded","fader","fades","fadge","fados","faena","faery","faffs","faffy","faggy","fagin","fagot","faiks","fails","faine","fains","faire","fairs","faked","faker","fakes","fakey","fakie","fakir","falaj","fales","falls","falsy","famed","fames","fanal","fands","fanes","fanga","fango","fangs","fanks","fanon","fanos","fanum","faqir","farad","farci","farcy","fards","fared","farer","fares","farle","farls","farms","faros","farro","farse","farts","fasci","fasti","fasts","fated","fates","fatly","fatso","fatwa","fauch","faugh","fauld","fauns","faurd","faute","fauts","fauve","favas","favel","faver","faves","favus","fawns","fawny","faxed","faxes","fayed","fayer","fayne","fayre","fazed","fazes","feals","feard","feare","fears","feart","fease","feats","feaze","feces","fecht","fecit","fecks","fedai","fedex","feebs","feeds","feels","feely","feens","feers","feese","feeze","fehme","feint","feist","felch","felid","felix","fells","felly","felts","felty","femal","femes","femic","femmy","fends","fendy","fenis","fenks","fenny","fents","feods","feoff","ferer","feres","feria","ferly","fermi","ferms","ferns","ferny","ferox","fesse","festa","fests","festy","fetas","feted","fetes","fetor","fetta","fetts","fetwa","feuar","feuds","feued","feyed","feyer","feyly","fezes","fezzy","fiars","fiats","fibre","fibro","fices","fiche","fichu","ficin","ficos","ficta","fides","fidge","fidos","fidus","fiefs","fient","fiere","fieri","fiers","fiest","fifed","fifer","fifes","fifis","figgy","figos","fiked","fikes","filar","filch","filed","files","filii","filks","fille","fillo","fills","filmi","films","filon","filos","filum","finca","finds","fined","fines","finis","finks","finny","finos","fiord","fiqhs","fique","fired","firer","fires","firie","firks","firma","firms","firni","firns","firry","firth","fiscs","fisho","fisks","fists","fisty","fitch","fitly","fitna","fitte","fitts","fiver","fives","fixed","fixes","fixie","fixit","fjeld","flabs","flaff","flags","flaks","flamm","flams","flamy","flane","flans","flaps","flary","flats","flava","flawn","flaws","flawy","flaxy","flays","fleam","fleas","fleek","fleer","flees","flegs","fleme","fleur","flews","flexi","flexo","fleys","flics","flied","flies","flimp","flims","flips","flirs","flisk","flite","flits","flitt","flobs","flocs","floes","flogs","flong","flops","flore","flors","flory","flosh","flota","flote","flows","flowy","flubs","flued","flues","fluey","fluky","flump","fluor","flurr","fluty","fluyt","flyby","flyin","flype","flyte","fnarr","foals","foams","foehn","fogey","fogie","fogle","fogos","fogou","fohns","foids","foils","foins","folds","foley","folia","folic","folie","folks","folky","fomes","fonda","fonds","fondu","fones","fonio","fonly","fonts","foods","foody","fools","foots","footy","foram","forbs","forby","fordo","fords","forel","fores","forex","forks","forky","forma","forme","forms","forts","forza","forze","fossa","fosse","fouat","fouds","fouer","fouet","foule","fouls","fount","fours","fouth","fovea","fowls","fowth","foxed","foxes","foxie","foyle","foyne","frabs","frack","fract","frags","fraim","frais","franc","frape","fraps","frass","frate","frati","frats","fraus","frays","frees","freet","freit","fremd","frena","freon","frere","frets","fribs","frier","fries","frigs","frise","frist","frita","frite","frith","frits","fritt","frize","frizz","froes","frogs","fromm","frons","froom","frore","frorn","frory","frosh","frows","frowy","froyo","frugs","frump","frush","frust","fryer","fubar","fubby","fubsy","fucks","fucus","fuddy","fudgy","fuels","fuero","fuffs","fuffy","fugal","fuggy","fugie","fugio","fugis","fugle","fugly","fugus","fujis","fulla","fulls","fulth","fulwa","fumed","fumer","fumes","fumet","funda","fundi","fundo","funds","fundy","fungo","fungs","funic","funis","funks","funsy","funts","fural","furan","furca","furls","furol","furos","furrs","furth","furze","furzy","fused","fusee","fusel","fuses","fusil","fusks","fusts","fusty","futon","fuzed","fuzee","fuzes","fuzil","fyces","fyked","fykes","fyles","fyrds","fytte","gabba","gabby","gable","gaddi","gades","gadge","gadgy","gadid","gadis","gadje","gadjo","gadso","gaffs","gaged","gager","gages","gaids","gains","gairs","gaita","gaits","gaitt","gajos","galah","galas","galax","galea","galed","gales","galia","galis","galls","gally","galop","galut","galvo","gamas","gamay","gamba","gambe","gambo","gambs","gamed","games","gamey","gamic","gamin","gamme","gammy","gamps","ganch","gandy","ganef","ganev","gangs","ganja","ganks","ganof","gants","gaols","gaped","gaper","gapes","gapos","gappy","garam","garba","garbe","garbo","garbs","garda","garde","gares","garis","garms","garni","garre","garri","garth","garum","gases","gashy","gasps","gaspy","gasts","gatch","gated","gater","gates","gaths","gator","gauch","gaucy","gauds","gauje","gault","gaums","gaumy","gaups","gaurs","gauss","gauzy","gavot","gawcy","gawds","gawks","gawps","gawsy","gayal","gazal","gazar","gazed","gazes","gazon","gazoo","geals","geans","geare","gears","geasa","geats","gebur","gecks","geeks","geeps","geest","geist","geits","gelds","gelee","gelid","gelly","gelts","gemel","gemma","gemmy","gemot","genae","genal","genas","genes","genet","genic","genii","genin","genio","genip","genny","genoa","genom","genro","gents","genty","genua","genus","geode","geoid","gerah","gerbe","geres","gerle","germs","germy","gerne","gesse","gesso","geste","gests","getas","getup","geums","geyan","geyer","ghast","ghats","ghaut","ghazi","ghees","ghest","ghusl","ghyll","gibed","gibel","giber","gibes","gibli","gibus","gifts","gigas","gighe","gigot","gigue","gilas","gilds","gilet","gilia","gills","gilly","gilpy","gilts","gimel","gimme","gimps","gimpy","ginch","ginga","ginge","gings","ginks","ginny","ginzo","gipon","gippo","gippy","girds","girlf","girls","girns","giron","giros","girrs","girsh","girts","gismo","gisms","gists","gitch","gites","giust","gived","gives","gizmo","glace","glads","glady","glaik","glair","glamp","glams","glans","glary","glatt","glaum","glaur","glazy","gleba","glebe","gleby","glede","gleds","gleed","gleek","glees","gleet","gleis","glens","glent","gleys","glial","glias","glibs","gliff","glift","glike","glime","glims","glisk","glits","glitz","gloam","globi","globs","globy","glode","glogg","gloms","gloop","glops","glost","glout","glows","glowy","gloze","glued","gluer","glues","gluey","glugg","glugs","glume","glums","gluon","glute","gluts","gnapi","gnarl","gnarr","gnars","gnats","gnawn","gnaws","gnows","goads","goafs","goaft","goals","goary","goats","goaty","goave","goban","gobar","gobbe","gobbi","gobbo","gobby","gobis","gobos","godet","godso","goels","goers","goest","goeth","goety","gofer","goffs","gogga","gogos","goier","gojis","gokes","golds","goldy","goles","golfs","golpe","golps","gombo","gomer","gompa","gonch","gonef","gongs","gonia","gonif","gonks","gonna","gonof","gonys","gonzo","gooby","goodo","goods","goofs","googs","gooks","gooky","goold","gools","gooly","goomy","goons","goony","goops","goopy","goors","goory","goosy","gopak","gopik","goral","goras","goray","gorbs","gordo","gored","gores","goris","gorms","gormy","gorps","gorse","gorsy","gosht","gosse","gotch","goths","gothy","gotta","gouch","gouks","goura","gouts","gouty","goved","goves","gowan","gowds","gowfs","gowks","gowls","gowns","goxes","goyim","goyle","graal","grabs","grads","graff","graip","grama","grame","gramp","grams","grana","grano","grans","grapy","grata","grats","gravs","grays","grebe","grebo","grece","greek","grees","grege","grego","grein","grens","greps","grese","greve","grews","greys","grice","gride","grids","griff","grift","grigs","grike","grins","griot","grips","gript","gripy","grise","grist","grisy","grith","grits","grize","groat","grody","grogs","groks","groma","groms","grone","groof","grosz","grots","grouf","grovy","grows","grrls","grrrl","grubs","grued","grues","grufe","grume","grump","grund","gryce","gryde","gryke","grype","grypt","guaco","guana","guano","guans","guars","gubba","gucks","gucky","gudes","guffs","gugas","guggl","guido","guids","guimp","guiro","gulab","gulag","gular","gulas","gules","gulet","gulfs","gulfy","gulls","gulph","gulps","gulpy","gumma","gummi","gumps","gunas","gundi","gundy","gunge","gungy","gunks","gunky","gunny","guqin","gurdy","gurge","gurks","gurls","gurly","gurns","gurry","gursh","gurus","gushy","gusla","gusle","gusli","gussy","gusts","gutsy","gutta","gutty","guyed","guyle","guyot","guyse","gwine","gyals","gyans","gybed","gybes","gyeld","gymps","gynae","gynie","gynny","gynos","gyoza","gypes","gypos","gyppo","gyppy","gyral","gyred","gyres","gyron","gyros","gyrus","gytes","gyved","gyver","gyves","haafs","haars","haats","hable","habus","hacek","hacks","hacky","hadal","haded","hades","hadji","hadst","haems","haere","haets","haffs","hafiz","hafta","hafts","haggs","haham","hahas","haick","haika","haiks","haiku","hails","haily","hains","haint","hairs","haith","hajes","hajis","hajji","hakam","hakas","hakea","hakes","hakim","hakus","halal","haldi","haled","haler","hales","halfa","halfs","halid","hallo","halls","halma","halms","halon","halos","halse","halsh","halts","halva","halwa","hamal","hamba","hamed","hamel","hames","hammy","hamza","hanap","hance","hanch","handi","hands","hangi","hangs","hanks","hanky","hansa","hanse","hants","haole","haoma","hapas","hapax","haply","happi","hapus","haram","hards","hared","hares","harim","harks","harls","harms","harns","haros","harps","harts","hashy","hasks","hasps","hasta","hated","hates","hatha","hathi","hatty","hauds","haufs","haugh","haugo","hauld","haulm","hauls","hault","hauns","hause","havan","havel","haver","haves","hawed","hawks","hawms","hawse","hayed","hayer","hayey","hayle","hazan","hazed","hazer","hazes","hazle","heads","heald","heals","heame","heaps","heapy","heare","hears","heast","heats","heaty","heben","hebes","hecht","hecks","heder","hedgy","heeds","heedy","heels","heeze","hefte","hefts","heiau","heids","heigh","heils","heirs","hejab","hejra","heled","heles","helio","hella","hells","helly","helms","helos","helot","helps","helve","hemal","hemes","hemic","hemin","hemps","hempy","hench","hends","henge","henna","henny","henry","hents","hepar","herbs","herby","herds","heres","herls","herma","herms","herns","heros","herps","herry","herse","hertz","herye","hesps","hests","hetes","heths","heuch","heugh","hevea","hevel","hewed","hewer","hewgh","hexad","hexed","hexer","hexes","hexyl","heyed","hiant","hibas","hicks","hided","hider","hides","hiems","hifis","highs","hight","hijab","hijra","hiked","hiker","hikes","hikoi","hilar","hilch","hillo","hills","hilsa","hilts","hilum","hilus","himbo","hinau","hinds","hings","hinky","hinny","hints","hiois","hiped","hiper","hipes","hiply","hired","hiree","hirer","hires","hissy","hists","hithe","hived","hiver","hives","hizen","hoach","hoaed","hoagy","hoars","hoary","hoast","hobos","hocks","hocus","hodad","hodja","hoers","hogan","hogen","hoggs","hoghs","hogoh","hogos","hohed","hoick","hoied","hoiks","hoing","hoise","hokas","hoked","hokes","hokey","hokis","hokku","hokum","holds","holed","holes","holey","holks","holla","hollo","holme","holms","holon","holos","holts","homas","homed","homes","homey","homie","homme","homos","honan","honda","honds","honed","honer","hones","hongi","hongs","honks","honky","hooch","hoods","hoody","hooey","hoofs","hoogo","hooha","hooka","hooks","hooky","hooly","hoons","hoops","hoord","hoors","hoosh","hoots","hooty","hoove","hopak","hoped","hoper","hopes","hoppy","horah","horal","horas","horis","horks","horme","horns","horst","horsy","hosed","hosel","hosen","hoser","hoses","hosey","hosta","hosts","hotch","hoten","hotis","hotte","hotty","houff","houfs","hough","houri","hours","houts","hovea","hoved","hoven","hoves","howay","howbe","howes","howff","howfs","howks","howls","howre","howso","howto","hoxed","hoxes","hoyas","hoyed","hoyle","hubba","hubby","hucks","hudna","hudud","huers","huffs","huffy","huger","huggy","huhus","huias","huies","hukou","hulas","hules","hulks","hulky","hullo","hulls","hully","humas","humfs","humic","humps","humpy","hundo","hunks","hunts","hurds","hurls","hurly","hurra","hurst","hurts","hurty","hushy","husks","husos","hutia","huzza","huzzy","hwyls","hydel","hydra","hyens","hygge","hying","hykes","hylas","hyleg","hyles","hylic","hymns","hynde","hyoid","hyped","hypes","hypha","hyphy","hypos","hyrax","hyson","hythe","iambi","iambs","ibrik","icers","iched","iches","ichor","icier","icker","ickle","icons","ictal","ictic","ictus","idant","iddah","iddat","iddut","ideas","idees","ident","idled","idles","idlis","idola","idols","idyls","iftar","igapo","igged","iglus","ignis","ihram","iiwis","ikans","ikats","ikons","ileac","ileal","ileum","ileus","iliad","ilial","ilium","iller","illth","imago","imagy","imams","imari","imaum","imbar","imbed","imbos","imide","imido","imids","imine","imino","imlis","immew","immit","immix","imped","impis","impot","impro","imshi","imshy","inapt","inarm","inbye","incas","incel","incle","incog","incus","incut","indew","india","indie","indol","indow","indri","indue","inerm","infix","infos","infra","ingan","ingle","inion","inked","inker","inkle","inned","innie","innit","inorb","inros","inrun","insee","inset","inspo","intel","intil","intis","intra","inula","inure","inurn","inust","invar","inver","inwit","iodic","iodid","iodin","ioras","iotas","ippon","irade","irids","iring","irked","iroko","irone","irons","isbas","ishes","isled","isles","isnae","issei","istle","items","ither","ivied","ivies","ixias","ixnay","ixora","ixtle","izard","izars","izzat","jaaps","jabot","jacal","jacet","jacks","jacky","jaded","jades","jafas","jaffa","jagas","jager","jaggs","jaggy","jagir","jagra","jails","jaker","jakes","jakey","jakie","jalap","jaleo","jalop","jambe","jambo","jambs","jambu","james","jammy","jamon","jamun","janes","janky","janns","janny","janty","japan","japed","japer","japes","jarks","jarls","jarps","jarta","jarul","jasey","jaspe","jasps","jatha","jatis","jatos","jauks","jaune","jaups","javas","javel","jawan","jawed","jawns","jaxie","jeans","jeats","jebel","jedis","jeels","jeely","jeeps","jeera","jeers","jeeze","jefes","jeffs","jehad","jehus","jelab","jello","jells","jembe","jemmy","jenny","jeons","jerid","jerks","jerry","jesse","jessy","jests","jesus","jetee","jetes","jeton","jeune","jewed","jewie","jhala","jheel","jhils","jiaos","jibba","jibbs","jibed","jiber","jibes","jiffs","jiggy","jigot","jihad","jills","jilts","jimmy","jimpy","jingo","jings","jinks","jinne","jinni","jinns","jirds","jirga","jirre","jisms","jitis","jitty","jived","jiver","jives","jivey","jnana","jobed","jobes","jocko","jocks","jocky","jocos","jodel","joeys","johns","joins","joked","jokes","jokey","jokol","joled","joles","jolie","jollo","jolls","jolts","jolty","jomon","jomos","jones","jongs","jonty","jooks","joram","jorts","jorum","jotas","jotty","jotun","joual","jougs","jouks","joule","jours","jowar","jowed","jowls","jowly","joyed","jubas","jubes","jucos","judas","judgy","judos","jugal","jugum","jujus","juked","jukes","jukus","julep","julia","jumar","jumby","jumps","junco","junks","junky","jupes","jupon","jural","jurat","jurel","jures","juris","juste","justs","jutes","jutty","juves","juvie","kaama","kabab","kabar","kabob","kacha","kacks","kadai","kades","kadis","kafir","kagos","kagus","kahal","kaiak","kaids","kaies","kaifs","kaika","kaiks","kails","kaims","kaing","kains","kajal","kakas","kakis","kalam","kalas","kales","kalif","kalis","kalpa","kalua","kamas","kames","kamik","kamis","kamme","kanae","kanal","kanas","kanat","kandy","kaneh","kanes","kanga","kangs","kanji","kants","kanzu","kaons","kapai","kapas","kapha","kaphs","kapok","kapow","kapur","kapus","kaput","karai","karas","karat","karee","karez","karks","karns","karoo","karos","karri","karst","karsy","karts","karzy","kasha","kasme","katal","katas","katis","katti","kaugh","kauri","kauru","kaury","kaval","kavas","kawas","kawau","kawed","kayle","kayos","kazis","kazoo","kbars","kcals","keaki","kebar","kebob","kecks","kedge","kedgy","keech","keefs","keeks","keels","keema","keeno","keens","keeps","keets","keeve","kefir","kehua","keirs","kelep","kelim","kells","kelly","kelps","kelpy","kelts","kelty","kembo","kembs","kemps","kempt","kempy","kenaf","kench","kendo","kenos","kente","kents","kepis","kerbs","kerel","kerfs","kerky","kerma","kerne","kerns","keros","kerry","kerve","kesar","kests","ketas","ketch","ketes","ketol","kevel","kevil","kexes","keyed","keyer","khadi","khads","khafs","khana","khans","khaph","khats","khaya","khazi","kheda","kheer","kheth","khets","khirs","khoja","khors","khoum","khuds","khula","khyal","kiaat","kiack","kiaki","kiang","kiasu","kibbe","kibbi","kibei","kibes","kibla","kicks","kicky","kiddo","kiddy","kidel","kideo","kidge","kiefs","kiers","kieve","kievs","kight","kikay","kikes","kikoi","kiley","kilig","kilim","kills","kilns","kilos","kilps","kilts","kilty","kimbo","kimet","kinas","kinda","kinds","kindy","kines","kings","kingy","kinin","kinks","kinos","kiore","kipah","kipas","kipes","kippa","kipps","kipsy","kirby","kirks","kirns","kirri","kisan","kissy","kists","kitab","kited","kiter","kites","kithe","kiths","kitke","kitul","kivas","kiwis","klang","klaps","klett","klick","klieg","kliks","klong","kloof","kluge","klutz","knags","knaps","knarl","knars","knaur","knawe","knees","knell","knick","knish","knits","knive","knobs","knoop","knops","knosp","knots","knoud","knout","knowd","knowe","knows","knubs","knule","knurl","knurr","knurs","knuts","koans","koaps","koban","kobos","koels","koffs","kofta","kogal","kohas","kohen","kohls","koine","koiwi","kojis","kokam","kokas","koker","kokra","kokum","kolas","kolos","kombi","kombu","konbu","kondo","konks","kooks","kooky","koori","kopek","kophs","kopje","koppa","korai","koran","koras","korat","kores","koris","korma","koros","korun","korus","koses","kotch","kotos","kotow","koura","kraal","krabs","kraft","krais","krait","krang","krans","kranz","kraut","krays","kreef","kreen","kreep","kreng","krewe","kriol","krona","krone","kroon","krubi","krump","krunk","ksars","kubie","kudos","kudus","kudzu","kufis","kugel","kuias","kukri","kukus","kulak","kulan","kulas","kulfi","kumis","kumys","kunas","kunds","kuris","kurre","kurta","kurus","kusso","kusti","kutai","kutas","kutch","kutis","kutus","kuyas","kuzus","kvass","kvell","kwaai","kwela","kwink","kwirl","kyack","kyaks","kyang","kyars","kyats","kybos","kydst","kyles","kylie","kylin","kylix","kyloe","kynde","kynds","kypes","kyrie","kytes","kythe","kyudo","laarf","laari","labda","labia","labis","labne","labra","laccy","laced","lacer","laces","lacet","lacey","lacis","lacka","lacks","lacky","laddu","laddy","laded","ladee","lader","lades","ladoo","laers","laevo","lagan","lagar","laggy","lahal","lahar","laich","laics","laide","laids","laigh","laika","laiks","laird","lairs","lairy","laith","laity","laked","laker","lakes","lakhs","lakin","laksa","laldy","lalls","lamas","lambs","lamby","lamed","lamer","lames","lamia","lammy","lamps","lanai","lanas","lanch","lande","lands","laned","lanes","lanks","lants","lapas","lapin","lapis","lapje","lappa","lappy","larch","lards","lardy","laree","lares","larfs","larga","largo","laris","larks","larky","larns","larnt","larum","lased","laser","lases","lassi","lassu","lassy","lasts","latah","lated","laten","latex","lathi","laths","lathy","latke","latus","lauan","lauch","laude","lauds","laufs","laund","laura","laval","lavas","laved","laver","laves","lavra","lavvy","lawed","lawer","lawin","lawks","lawns","lawny","lawsy","laxed","laxer","laxes","laxly","layby","layed","layin","layup","lazar","lazed","lazes","lazos","lazzi","lazzo","leads","leady","leafs","leaks","leams","leans","leany","leaps","leare","lears","leary","leats","leavy","leaze","leben","leccy","leche","ledes","ledgy","ledum","leear","leeks","leeps","leers","leese","leets","leeze","lefte","lefts","leger","leges","legge","leggo","legit","legno","lehrs","lehua","leirs","leish","leman","lemed","lemel","lemes","lemma","lemme","lends","lenes","lengs","lenis","lenos","lense","lenti","lento","leone","lepak","lepid","lepra","lepta","lered","leres","lerps","lesbo","leses","lesos","lests","letch","lethe","letty","letup","leuch","leuco","leuds","leugh","levas","levee","leves","levin","levis","lewis","lexes","lexis","lezes","lezza","lezzo","lezzy","liana","liane","liang","liard","liars","liart","liber","libor","libra","libre","libri","licet","lichi","licht","licit","licks","lidar","lidos","liefs","liens","liers","lieus","lieve","lifer","lifes","lifey","lifts","ligan","liger","ligge","ligne","liked","liker","likes","likin","lills","lilos","lilts","lilty","liman","limas","limax","limba","limbi","limbs","limby","limed","limen","limes","limey","limma","limns","limos","limpa","limps","linac","linch","linds","lindy","lined","lines","liney","linga","lings","lingy","linin","links","linky","linns","linny","linos","lints","linty","linum","linux","lions","lipas","lipes","lipin","lipos","lippy","liras","lirks","lirot","lises","lisks","lisle","lisps","lists","litai","litas","lited","litem","liter","lites","litho","liths","litie","litre","lived","liven","lives","livor","livre","liwaa","liwas","llano","loach","loads","loafs","loams","loans","loast","loave","lobar","lobed","lobes","lobos","lobus","loche","lochs","lochy","locie","locis","locks","locky","locos","locum","loden","lodes","loess","lofts","logan","loges","loggy","logia","logie","logoi","logon","logos","lohan","loids","loins","loipe","loirs","lokes","lokey","lokum","lolas","loled","lollo","lolls","lolly","lolog","lolos","lomas","lomed","lomes","loner","longa","longe","longs","looby","looed","looey","loofa","loofs","looie","looks","looky","looms","loons","loony","loops","loord","loots","loped","loper","lopes","loppy","loral","loran","lords","lordy","lorel","lores","loric","loris","losed","losel","losen","loses","lossy","lotah","lotas","lotes","lotic","lotos","lotsa","lotta","lotte","lotto","lotus","loued","lough","louie","louis","louma","lound","louns","loupe","loups","loure","lours","loury","louts","lovat","loved","lovee","loves","lovey","lovie","lowan","lowed","lowen","lowes","lownd","lowne","lowns","lowps","lowry","lowse","lowth","lowts","loxed","loxes","lozen","luach","luaus","lubed","lubes","lubra","luces","lucks","lucre","ludes","ludic","ludos","luffa","luffs","luged","luger","luges","lulls","lulus","lumas","lumbi","lumme","lummy","lumps","lunas","lunes","lunet","lungi","lungs","lunks","lunts","lupin","lured","lurer","lures","lurex","lurgi","lurgy","lurks","lurry","lurve","luser","lushy","lusks","lusts","lusus","lutea","luted","luter","lutes","luvvy","luxed","luxer","luxes","lweis","lyams","lyard","lyart","lyase","lycea","lycee","lycra","lymes","lynch","lynes","lyres","lysed","lyses","lysin","lysis","lysol","lyssa","lyted","lytes","lythe","lytic","lytta","maaed","maare","maars","maban","mabes","macas","macca","maced","macer","maces","mache","machi","machs","macka","macks","macle","macon","macte","madal","madar","maddy","madge","madid","mados","madre","maedi","maerl","mafic","mafts","magas","mages","maggs","magna","magot","magus","mahal","mahem","mahis","mahoe","mahrs","mahua","mahwa","maids","maiko","maiks","maile","maill","mailo","mails","maims","mains","maire","mairs","maise","maist","majas","majat","majoe","majos","makaf","makai","makan","makar","makee","makes","makie","makis","makos","malae","malai","malam","malar","malas","malax","maleo","males","malic","malik","malis","malky","malls","malms","malmy","malts","malty","malus","malva","malwa","mamak","mamas","mamba","mambu","mamee","mamey","mamie","mamil","manas","manat","mandi","mands","mandy","maneb","maned","maneh","manes","manet","mangi","mangs","manie","manis","manks","manky","manna","manny","manoa","manos","manse","manso","manta","mante","manto","mants","manty","manul","manus","manzo","mapau","mapes","mapou","mappy","maqam","maqui","marae","marah","maral","maran","maras","maray","marcs","mards","mardy","mares","marga","marge","margo","margs","maria","marid","maril","marka","marks","marle","marls","marly","marma","marms","maron","maror","marra","marri","marse","marts","marua","marvy","masas","mased","maser","mases","masha","mashy","masks","massa","massy","masts","masty","masur","masus","masut","matai","mated","mater","mates","mathe","maths","matin","matlo","matra","matsu","matte","matts","matty","matza","matzo","mauby","mauds","mauka","maula","mauls","maums","maumy","maund","maunt","mauri","mausy","mauts","mauvy","mauzy","maven","mavie","mavin","mavis","mawed","mawks","mawky","mawla","mawns","mawps","mawrs","maxed","maxes","maxis","mayan","mayas","mayed","mayos","mayst","mazac","mazak","mazar","mazas","mazed","mazel","mazer","mazes","mazet","mazey","mazut","mbari","mbars","mbila","mbira","mbret","mbube","mbuga","meads","meake","meaks","meals","meane","means","meany","meare","mease","meath","meats","mebbe","mebos","mecha","mechs","mecks","mecum","medii","medin","medle","meech","meeds","meeja","meeps","meers","meets","meffs","meids","meiko","meils","meins","meint","meiny","meism","meith","mekka","melam","melas","melba","melch","melds","meles","melic","melik","mells","meloe","melos","melts","melty","memes","memic","memos","menad","mence","mends","mened","menes","menge","mengs","menil","mensa","mense","mensh","menta","mento","ments","menus","meous","meows","merch","mercs","merde","merds","mered","merel","merer","meres","meril","meris","merks","merle","merls","merse","mersk","mesad","mesal","mesas","mesca","mesel","mesem","meses","meshy","mesia","mesic","mesne","meson","messy","mesto","mesyl","metas","meted","meteg","metel","metes","methi","metho","meths","methy","metic","metif","metis","metol","metre","metta","meums","meuse","meved","meves","mewed","mewls","meynt","mezes","mezza","mezze","mezzo","mgals","mhorr","miais","miaou","miaow","miasm","miaul","micas","miche","michi","micht","micks","micky","micos","micra","middy","midgy","midis","miens","mieux","mieve","miffs","miffy","mifty","miggs","migma","migod","mihas","mihis","mikan","miked","mikes","mikos","mikra","mikva","milch","milds","miler","miles","milfs","milia","milko","milks","mille","mills","milly","milor","milos","milpa","milts","milty","miltz","mimed","mimeo","mimer","mimes","mimis","mimsy","minae","minar","minas","mincy","mindi","minds","mined","mines","minge","mingi","mings","mingy","minis","minke","minks","minny","minos","minse","mints","minxy","miraa","mirah","mirch","mired","mires","mirex","mirid","mirin","mirkn","mirks","mirky","mirls","mirly","miros","mirrl","mirrs","mirvs","mirza","misal","misch","misdo","mises","misgo","misky","misls","misos","missa","misto","mists","misty","mitas","mitch","miter","mites","mitey","mitie","mitis","mitre","mitry","mitta","mitts","mivey","mivvy","mixed","mixen","mixer","mixes","mixie","mixis","mixte","mixup","miyas","mizen","mizes","mizzy","mmkay","mneme","moais","moaky","moals","moana","moans","moany","moars","moats","mobby","mobed","mobee","mobes","mobey","mobie","moble","mobos","mocap","mochi","mochs","mochy","mocks","mocky","mocos","mocus","moder","modes","modge","modii","modin","modoc","modom","modus","moeni","moers","mofos","mogar","mogas","moggy","mogos","mogra","mogue","mohar","mohel","mohos","mohrs","mohua","mohur","moile","moils","moira","moire","moits","moity","mojos","moker","mokes","mokey","mokis","mokky","mokos","mokus","molal","molas","molds","moled","moler","moles","moley","molie","molla","molle","mollo","molls","molly","moloi","molos","molto","molts","molue","molvi","molys","momes","momie","momma","momme","mommy","momos","mompe","momus","monad","monal","monas","monde","mondo","moner","mongo","mongs","monic","monie","monks","monos","monpe","monte","monty","moobs","mooch","moods","mooed","mooey","mooks","moola","mooli","mools","mooly","moong","mooni","moons","moony","moops","moors","moory","mooth","moots","moove","moped","moper","mopes","mopey","moppy","mopsy","mopus","morae","morah","moran","moras","morat","moray","moree","morel","mores","morgy","moria","morin","mormo","morna","morne","morns","moror","morra","morro","morse","morts","moruk","mosed","moses","mosey","mosks","mosso","moste","mosto","mosts","moted","moten","motes","motet","motey","moths","mothy","motis","moton","motte","motts","motty","motus","motza","mouch","moues","moufs","mould","moule","mouls","mouly","moups","moust","mousy","moved","moves","mowas","mowed","mowie","mowra","moxas","moxie","moyas","moyle","moyls","mozed","mozes","mozos","mpret","mrads","msasa","mtepe","mucho","mucic","mucid","mucin","mucko","mucks","mucor","mucro","mudar","mudge","mudif","mudim","mudir","mudra","muffs","muffy","mufti","mugga","muggs","muggy","mugho","mugil","mugos","muhly","muids","muils","muirs","muiry","muist","mujik","mukim","mukti","mulai","mulct","muled","mules","muley","mulga","mulie","mulla","mulls","mulse","mulsh","mumbo","mumms","mumph","mumps","mumsy","mumus","munds","mundu","munga","munge","mungi","mungo","mungs","mungy","munia","munis","munja","munjs","munts","muntu","muons","muras","mured","mures","murex","murgh","murgi","murid","murks","murls","murly","murra","murre","murri","murrs","murry","murth","murti","muruk","murva","musar","musca","mused","musee","muser","muses","muset","musha","musit","musks","musos","musse","mussy","musta","musth","musts","mutas","mutch","muted","muter","mutes","mutha","mutic","mutis","muton","mutti","mutts","mutum","muvva","muxed","muxes","muzak","muzzy","mvula","mvule","mvuli","myall","myals","mylar","mynah","mynas","myoid","myoma","myons","myope","myops","myopy","mysid","mysie","mythi","myths","mythy","myxos","mzees","naams","naans","naats","nabam","nabby","nabes","nabis","nabks","nabla","nabob","nache","nacho","nacre","nadas","naeve","naevi","naffs","nagar","nagas","nages","naggy","nagor","nahal","naiad","naibs","naice","naids","naieo","naifs","naiks","nails","naily","nains","naios","naira","nairu","najib","nakas","naked","naker","nakfa","nalas","naled","nalla","namad","namak","namaz","named","namer","names","namma","namus","nanas","nance","nancy","nandu","nanna","nanos","nante","nanti","nanto","nants","nanty","nanua","napas","naped","napes","napoh","napoo","nappa","nappe","nappy","naras","narco","narcs","nards","nares","naric","naris","narks","narky","narod","narra","narre","nashi","nasho","nasis","nason","nasus","natak","natch","nates","natis","natto","natty","natya","nauch","naunt","navar","naved","naves","navew","navvy","nawab","nawal","nazar","nazes","nazir","nazis","nazzy","nduja","neafe","neals","neant","neaps","nears","neath","neato","neats","nebby","nebek","nebel","neche","necks","neddy","neebs","needs","neefs","neeld","neele","neemb","neems","neeps","neese","neeze","nefie","negri","negro","negus","neifs","neist","neive","nelia","nelis","nelly","nemas","nemic","nemns","nempt","nenes","nenta","neons","neosa","neoza","neper","nepit","neral","neram","nerds","nerfs","nerka","nerks","nerol","nerts","nertz","nervy","neski","nests","nesty","netas","netes","netop","netta","netts","netty","neuks","neume","neums","nevel","neves","nevis","nevus","nevvy","newbs","newed","newel","newie","newsy","newts","nexal","nexin","nexts","nexum","nexus","ngaio","ngaka","ngana","ngapi","ngati","ngege","ngoma","ngoni","ngram","ngwee","nibby","nicad","niced","nicey","nicht","nicks","nicky","nicol","nidal","nided","nides","nidor","nidus","niefs","niess","nieve","nifes","niffs","niffy","nifle","nifty","niger","nigga","nighs","nigre","nigua","nihil","nikab","nikah","nikau","nilas","nills","nimbi","nimbs","nimby","nimps","niner","nines","ninon","ninta","niopo","nioza","nipas","nipet","nippy","niqab","nirls","nirly","nisei","nisin","nisse","nisus","nital","niter","nites","nitid","niton","nitre","nitro","nitry","nitta","nitto","nitty","nival","nivas","nivel","nixed","nixer","nixes","nixie","nizam","njirl","nkosi","nmoli","nmols","noahs","nobby","nocks","nodal","noddy","noded","nodes","nodum","nodus","noels","noema","noeme","nogal","noggs","noggy","nohow","noias","noils","noily","noint","noire","noirs","nokes","noles","nolle","nolls","nolos","nomas","nomen","nomes","nomic","nomoi","nomos","nonan","nonas","nonce","noncy","nonda","nondo","nones","nonet","nongs","nonic","nonis","nonna","nonno","nonny","nonyl","noobs","noois","nooit","nooks","nooky","noone","noons","noops","noove","nopal","noria","norie","noris","norks","norma","norms","nosed","noser","noses","noshi","nosir","notal","notam","noted","noter","notes","notum","nougs","nouja","nould","noule","nouls","nouns","nouny","noups","noust","novae","novas","novia","novio","novum","noway","nowds","nowed","nowls","nowts","nowty","noxal","noxas","noxes","noyau","noyed","noyes","nrtta","nrtya","nsima","nubby","nubia","nucha","nucin","nuddy","nuder","nudes","nudgy","nudie","nudzh","nuevo","nuffs","nugae","nujol","nuked","nukes","nulla","nullo","nulls","nully","numbs","numen","nummy","numps","nunks","nunky","nunny","nunus","nuque","nurds","nurdy","nurls","nurrs","nurts","nurtz","nused","nuses","nutso","nutsy","nyaff","nyala","nyams","nying","nyong","nyssa","nyung","nyuse","nyuze","oafos","oaked","oaker","oakum","oared","oarer","oasal","oases","oasis","oasts","oaten","oater","oaths","oaves","obang","obbos","obeah","obeli","obeys","obias","obied","obiit","obits","objet","oboes","obole","oboli","obols","occam","ocher","oches","ochre","ochry","ocker","ocote","ocrea","octad","octan","octas","octic","octli","octyl","oculi","odahs","odals","odeon","odeum","odism","odist","odium","odoom","odors","odour","odums","odyle","odyls","ofays","offed","offie","oflag","ofter","ofuro","ogams","ogeed","ogees","oggin","ogham","ogive","ogled","ogler","ogles","ogmic","ogres","ohelo","ohias","ohing","ohmic","ohone","oicks","oidia","oiled","oiler","oilet","oinks","oints","oiran","ojime","okapi","okays","okehs","okies","oking","okole","okras","okrug","oktas","olate","oldie","oldly","olehs","oleic","olein","olent","oleos","oleum","oleyl","oligo","olios","oliva","ollas","ollav","oller","ollie","ology","olona","olpae","olpes","omasa","omber","ombus","omdah","omdas","omdda","omdeh","omees","omens","omers","omiai","omits","omlah","ommel","ommin","omnes","omovs","omrah","omuls","oncer","onces","oncet","oncus","ondes","ondol","onely","oners","onery","ongon","onium","onkus","onlap","onlay","onmun","onned","onsen","ontal","ontic","ooaas","oobit","oohed","ooids","oojah","oomph","oonts","oopak","ooped","oopsy","oorie","ooses","ootid","ooyah","oozed","oozes","oozie","oozle","opahs","opals","opens","opepe","opery","opgaf","opihi","oping","oppos","opsat","opsin","opsit","opted","opter","opzit","orach","oracy","orals","orang","orans","orant","orate","orbat","orbed","orbic","orcas","orcin","ordie","ordos","oread","orfes","orful","orgia","orgic","orgue","oribi","oriel","origo","orixa","orles","orlon","orlop","ormer","ornee","ornis","orped","orpin","orris","ortet","ortho","orval","orzos","osars","oscar","osetr","oseys","oshac","osier","oskin","oslin","osmic","osmol","osone","ossia","ostia","otaku","otary","othyl","otium","ottar","ottos","oubit","ouche","oucht","oueds","ouens","ouija","oulks","oumas","oundy","oupas","ouped","ouphe","ouphs","ourey","ourie","ousel","ousia","ousts","outby","outed","outen","outie","outre","outro","outta","ouzel","ouzos","ovals","ovels","ovens","overs","ovism","ovist","ovoli","ovolo","ovule","oware","owari","owche","owers","owies","owled","owler","owlet","owned","ownio","owres","owrie","owsen","oxbow","oxeas","oxers","oxeye","oxids","oxies","oxime","oxims","oxine","oxlip","oxman","oxmen","oxter","oyama","oyers","ozeki","ozena","ozzie","paaho","paals","paans","pacai","pacas","pacay","paced","pacer","paces","pacey","pacha","packs","packy","pacos","pacta","pacts","padam","padas","paddo","padis","padle","padma","padou","padre","padri","paean","paedo","paeon","paged","pager","pages","pagle","pagne","pagod","pagri","pahit","pahos","pahus","paiks","pails","pains","paipe","paips","paire","pairs","paisa","paise","pakay","pakka","pakki","pakua","pakul","palak","palar","palas","palay","palea","paled","pales","palet","palis","palki","palla","palls","pallu","pally","palms","palmy","palpi","palps","palsa","palus","pamby","pampa","panax","pance","panch","panda","pands","pandy","paned","panes","panga","pangs","panim","panir","panko","panks","panna","panne","panni","panny","panto","pants","panty","paoli","paolo","papad","papas","papaw","papes","papey","pappi","pappy","papri","parae","paras","parch","parcs","pardi","pards","pardy","pared","paren","pareo","pares","pareu","parev","parge","pargo","parid","paris","parki","parks","parky","parle","parly","parma","parmo","parms","parol","parps","parra","parrs","parte","parti","parts","parve","parvo","pasag","pasar","pasch","paseo","pases","pasha","pashm","paska","pasmo","paspy","passe","passu","pasts","patas","pated","patee","patel","paten","pater","pates","paths","patia","patin","patka","patly","patta","patte","pattu","patus","pauas","pauls","pauxi","pavan","pavas","paved","paven","paver","paves","pavid","pavie","pavin","pavis","pavon","pavvy","pawas","pawaw","pawed","pawer","pawks","pawky","pawls","pawns","paxes","payed","payor","paysd","peage","peags","peake","peaks","peaky","peals","peans","peare","pears","peart","pease","peasy","peats","peaty","peavy","peaze","pebas","pechs","pecia","pecke","pecks","pecky","pects","pedes","pedis","pedon","pedos","pedro","peece","peeks","peeky","peels","peely","peens","peent","peeoy","peepe","peeps","peepy","peers","peery","peeve","peevo","peggy","peghs","pegma","pegos","peine","peins","peise","peisy","peize","pekan","pekau","pekea","pekes","pekid","pekin","pekoe","pelas","pelau","pelch","peles","pelfs","pells","pelma","pelog","pelon","pelsh","pelta","pelts","pelus","pends","pendu","pened","penes","pengo","penie","penis","penks","penna","penni","pense","pensy","pents","peola","peons","peony","pepla","peple","pepon","pepos","peppy","pepsi","pequi","perae","perai","perce","percs","perdu","perdy","perea","peres","perfs","peris","perks","perle","perls","perms","permy","perne","perns","perog","perps","perry","perse","persp","perst","perts","perve","pervo","pervs","pervy","pesch","pesos","pesta","pests","pesty","petar","peter","petit","petos","petre","petri","petti","petto","pewed","pewee","pewit","peyse","pfftt","phage","phang","phare","pharm","phasm","pheer","pheme","phene","pheon","phese","phial","phies","phish","phizz","phlox","phobe","phoca","phono","phons","phooh","phooo","phota","phots","photy","phpht","phubs","phuts","phutu","phwat","phyla","phyle","phyma","phynx","physa","piais","piani","pians","pibal","pical","picas","piccy","picey","pichi","picks","picon","picot","picra","picul","pieds","piend","piers","piert","pieta","piets","piezo","pight","pigly","pigmy","piing","pikas","pikau","piked","pikel","piker","pikes","pikey","pikis","pikul","pilae","pilaf","pilao","pilar","pilau","pilaw","pilch","pilea","piled","pilei","piler","piles","piley","pilin","pilis","pills","pilon","pilow","pilum","pilus","pimas","pimps","pinas","pinax","pince","pinda","pinds","pined","piner","pines","pinga","pinge","pingo","pings","pinko","pinks","pinna","pinny","pinol","pinon","pinot","pinta","pints","pinup","pions","piony","pious","pioye","pioys","pipal","pipas","piped","pipes","pipet","pipid","pipis","pipit","pippy","pipul","piqui","pirai","pirks","pirls","pirns","pirog","pirre","pirri","pirrs","pisco","pises","pisky","pisos","pissy","piste","pitas","piths","piton","pitot","pitso","pitsu","pitta","pittu","piuma","piums","pivos","pixes","piyut","pized","pizer","pizes","plaas","plack","plaga","plage","plaig","planc","planh","plans","plaps","plash","plasm","plast","plats","platt","platy","plaud","plaur","plavs","playa","plays","pleas","plebe","plebs","pleck","pleep","plein","plena","plene","pleno","pleon","plesh","plets","plews","plexi","plica","plies","pligs","plims","pling","plink","plips","plish","ploat","ploce","plock","plods","ploit","plomb","plong","plonk","plook","ploot","plops","plore","plots","plotz","plouk","plout","plows","plowt","ploye","ploys","pluds","plues","pluff","plugs","pluke","plums","plumy","plung","pluot","plups","plute","pluto","pluty","plyer","pneus","poach","poaka","poake","poalo","pobby","poboy","pocan","poche","pocho","pocks","pocky","podal","poddy","podex","podge","podgy","podia","podos","podus","poems","poena","poeps","poete","poets","pogey","pogge","poggy","pogos","pogue","pohed","poilu","poind","poire","pokal","poked","pokes","pokey","pokie","pokit","poled","poler","poles","poley","polio","polis","polje","polks","pollo","polls","polly","polos","polts","polys","pomas","pombe","pomes","pomme","pommy","pomos","pompa","pomps","ponce","poncy","ponds","pondy","pones","poney","ponga","pongo","pongs","pongy","ponks","ponor","ponto","ponts","ponty","ponzu","pooay","poods","pooed","pooey","poofs","poofy","poohs","poohy","pooja","pooka","pooks","pools","pooly","poons","poopa","poops","poopy","poori","poort","poots","pooty","poove","poovy","popes","popia","popos","poppa","popsy","popup","porae","poral","pored","porer","pores","porey","porge","porgy","porin","porks","porky","porno","porns","porny","porta","porte","porth","ports","porty","porus","posca","posed","poses","poset","posey","posho","posol","poste","posts","potae","potai","potch","poted","potes","potin","potoo","potro","potsy","potto","potts","potty","pouce","pouff","poufs","poufy","pouis","pouke","pouks","poule","poulp","poult","poupe","poupt","pours","pousy","pouts","povos","powan","powie","powin","powis","powlt","pownd","powns","powny","powre","powsy","poxed","poxes","poyas","poynt","poyou","poyse","pozzy","praam","prads","prags","prahu","prams","prana","prang","praos","praps","prase","prate","prats","pratt","praty","praus","prays","preak","predy","preed","preem","prees","preif","preke","prems","premy","prent","preon","preop","preps","presa","prese","prest","preta","preux","preve","prexy","preys","prial","prian","pricy","pridy","prief","prier","pries","prigs","prill","prima","primi","primp","prims","primy","pring","prink","prion","prise","priss","prius","proal","proas","probs","proby","prodd","prods","proem","profs","progs","proin","proke","prole","proll","promo","proms","pronk","prook","proot","props","prora","prore","proso","pross","prost","prosy","proto","proul","prowk","prows","proyn","pruno","prunt","pruny","pruta","pryan","pryer","pryse","pseud","pshaw","pshut","psias","psion","psoae","psoai","psoas","psora","psych","psyop","ptish","ptype","pubby","pubco","pubes","pubis","pubsy","pucan","pucer","puces","pucka","pucks","puddy","pudge","pudic","pudor","pudsy","pudus","puers","puffa","puffs","puggy","pugil","puhas","pujah","pujas","pukas","puked","puker","pukes","pukey","pukka","pukus","pulao","pulas","puled","puler","pules","pulik","pulis","pulka","pulks","pulli","pulls","pully","pulmo","pulps","pulus","pulut","pumas","pumie","pumps","pumpy","punas","punce","punga","pungi","pungo","pungs","pungy","punim","punji","punka","punks","punky","punny","punto","punts","punty","pupae","pupal","pupas","puppa","pupus","purao","purau","purda","purdy","pured","pures","purga","purin","puris","purls","puros","purps","purpy","purre","purrs","purry","pursy","purty","puses","pusle","pussy","putas","puter","putid","putin","puton","putos","putti","putto","putts","puttu","putza","puuko","puyas","puzel","puzta","pwned","pyats","pyets","pygal","pyins","pylon","pyned","pynes","pyoid","pyots","pyral","pyran","pyres","pyrex","pyric","pyros","pyrus","pyuff","pyxed","pyxes","pyxie","pyxis","pzazz","qadis","qaids","qajaq","qanat","qapik","qibla","qilas","qipao","qophs","qorma","quabs","quads","quaff","quags","quair","quais","quaky","quale","qualy","quank","quant","quare","quarl","quass","quate","quats","quawk","quaws","quayd","quays","qubit","quean","queck","queek","queem","queme","quena","quern","queso","quete","queyn","queys","queyu","quibs","quich","quids","quies","quiff","quila","quims","quina","quine","quink","quino","quins","quint","quipo","quips","quipu","quire","quirl","quirt","quist","quits","quoad","quods","quoif","quoin","quois","quoit","quoll","quonk","quops","quork","quorl","quouk","quoys","quran","qursh","quyte","raads","raake","rabat","rabic","rabis","raced","races","rache","racks","racon","raddi","raddy","radge","radgy","radif","radix","radon","rafee","raffs","raffy","rafik","rafiq","rafts","rafty","ragas","ragde","raged","ragee","rager","rages","ragga","raggs","raggy","ragis","ragus","rahed","rahui","raiah","raias","raids","raike","raiks","raile","rails","raine","rains","raird","raita","raith","raits","rajas","rajes","raked","rakee","raker","rakes","rakhi","rakia","rakis","rakki","raksi","rakus","rales","ralli","ramal","ramee","rames","ramet","ramie","ramin","ramis","rammy","ramon","ramps","ramse","ramsh","ramus","ranas","rance","rando","rands","raned","ranee","ranes","ranga","rangi","rangs","rangy","ranid","ranis","ranke","ranks","ranns","ranny","ranse","rants","ranty","raped","rapee","raper","rapes","raphe","rapin","rappe","rapso","rared","raree","rares","rarks","rasam","rasas","rased","raser","rases","rasps","rasse","rasta","ratal","ratan","ratas","ratch","rated","ratel","rater","rates","ratha","rathe","raths","ratoo","ratos","ratti","ratus","rauli","rauns","raupo","raved","ravel","raver","raves","ravey","ravin","rawdy","rawer","rawin","rawks","rawly","rawns","raxed","raxes","rayah","rayas","rayed","rayle","rayls","rayne","razai","razed","razee","razer","razes","razet","razoo","readd","reads","reais","reaks","realo","reals","reame","reams","reamy","reans","reaps","reard","rears","reast","reata","reate","reave","rebab","rebbe","rebec","rebid","rebit","rebop","rebud","rebuy","recal","recce","recco","reccy","recep","recit","recks","recon","recta","recte","recti","recto","recue","redan","redds","reddy","reded","redes","redia","redid","redif","redig","redip","redly","redon","redos","redox","redry","redub","redug","redux","redye","reeaf","reech","reede","reeds","reefs","reefy","reeks","reeky","reels","reely","reems","reens","reerd","reest","reeve","reeze","refan","refed","refel","reffo","refis","refix","refly","refry","regar","reges","reget","regex","reggo","regia","regie","regle","regma","regna","regos","regot","regur","rehem","reifs","reify","reiki","reiks","reine","reing","reink","reins","reird","reist","reive","rejas","rejig","rejon","reked","rekes","rekey","relet","relie","relit","rello","relos","reman","remap","remen","remet","remex","remix","remou","renay","rends","rendu","reney","renga","rengs","renig","renin","renks","renne","renos","rente","rents","reoil","reorg","repas","repat","repeg","repen","repin","repla","repos","repot","repps","repro","repun","reput","reran","rerig","resam","resat","resaw","resay","resee","reses","resew","resid","resit","resod","resol","resow","resto","rests","resty","resue","resus","retag","retam","retax","retem","retia","retie","retin","retip","retox","reune","reups","revet","revie","revow","rewan","rewax","rewed","rewet","rewin","rewon","rewth","rexes","rezes","rhabd","rheas","rheid","rheme","rheum","rhies","rhime","rhine","rhody","rhomb","rhone","rhumb","rhymy","rhyne","rhyta","riads","rials","riant","riata","riato","ribas","ribby","ribes","riced","ricer","rices","ricey","riche","richt","ricin","ricks","rides","ridgy","ridic","riels","riems","rieve","rifer","riffs","riffy","rifte","rifts","rifty","riggs","rigmo","rigol","rikka","rikwa","riled","riles","riley","rille","rills","rilly","rimae","rimed","rimer","rimes","rimon","rimus","rince","rinds","rindy","rines","ringe","rings","ringy","rinks","rioja","rione","riots","rioty","riped","ripes","ripps","riqqs","rises","rishi","risks","risps","rists","risus","rites","rithe","ritts","ritzy","rivas","rived","rivel","riven","rives","riyal","rizas","roads","roady","roake","roaky","roams","roans","roany","roars","roary","roate","robbo","robed","rober","robes","roble","robug","robur","roche","rocks","roded","rodes","rodny","roers","rogan","roguy","rohan","rohes","rohun","rohus","roids","roils","roily","roins","roist","rojak","rojis","roked","roker","rokes","rokey","rokos","rolag","roleo","roles","rolfs","rolls","rolly","romal","roman","romeo","romer","romps","rompu","rompy","ronde","rondo","roneo","rones","ronin","ronne","ronte","ronts","ronuk","roods","roofs","roofy","rooks","rooky","rooms","roons","roops","roopy","roosa","roose","roots","rooty","roped","roper","ropes","ropey","roque","roral","rores","roric","rorid","rorie","rorts","rorty","rosal","rosco","rosed","roses","roset","rosha","roshi","rosin","rosit","rosps","rossa","rosso","rosti","rosts","rotal","rotan","rotas","rotch","roted","rotes","rotis","rotls","roton","rotos","rotta","rotte","rotto","rotty","rouen","roues","rouet","roufs","rougy","rouks","rouky","roule","rouls","roums","roups","roupy","roust","routh","routs","roved","roven","roves","rowan","rowed","rowel","rowen","rowet","rowie","rowme","rownd","rowns","rowth","rowts","royet","royne","royst","rozes","rozet","rozit","ruach","ruana","rubai","ruban","rubby","rubel","rubes","rubin","rubio","ruble","rubli","rubor","rubus","ruche","ruchy","rucks","rudas","rudds","rudes","rudie","rudis","rueda","ruers","ruffe","ruffs","ruffy","rufus","rugae","rugal","rugas","ruggy","ruice","ruing","ruins","rukhs","ruled","rules","rully","rumal","rumbo","rumen","rumes","rumly","rummy","rumpo","rumps","rumpy","runce","runch","runds","runed","runer","runes","rungs","runic","runny","runos","runts","runty","runup","ruote","rupia","rurps","rurus","rusas","ruses","rushy","rusks","rusky","rusma","russe","rusts","ruths","rutin","rutty","ruvid","ryals","rybat","ryiji","ryijy","ryked","rykes","rymer","rymme","rynds","ryoti","ryots","ryper","rypin","rythe","ryugi","saags","sabal","sabed","saber","sabes","sabha","sabin","sabir","sabji","sable","sabos","sabot","sabra","sabre","sabzi","sacks","sacra","sacre","saddo","saddy","sades","sadhe","sadhu","sadic","sadis","sados","sadza","saeta","safed","safes","sagar","sagas","sager","sages","saggy","sagos","sagum","sahab","saheb","sahib","saice","saick","saics","saids","saiga","sails","saims","saine","sains","sairs","saist","saith","sajou","sakai","saker","sakes","sakia","sakis","sakti","salal","salas","salat","salep","sales","salet","salic","salis","salix","salle","salmi","salol","salop","salpa","salps","salse","salto","salts","salud","salue","salut","saman","samas","samba","sambo","samek","samel","samen","sames","samey","samfi","samfu","sammy","sampi","samps","sanad","sands","saned","sanes","sanga","sangh","sango","sangs","sanko","sansa","santo","sants","saola","sapan","sapid","sapor","saran","sards","sared","saree","sarge","sargo","sarin","sarir","saris","sarks","sarky","sarod","saros","sarus","sarvo","saser","sasin","sasse","satai","satay","sated","satem","sater","sates","satis","sauba","sauch","saugh","sauls","sault","saunf","saunt","saury","sauts","sauve","saved","saver","saves","savey","savin","sawah","sawed","sawer","saxes","sayas","sayed","sayee","sayer","sayid","sayne","sayon","sayst","sazes","scabs","scads","scaff","scags","scail","scala","scall","scams","scand","scans","scapa","scape","scapi","scarp","scars","scart","scath","scats","scatt","scaud","scaup","scaur","scaws","sceat","scena","scend","schav","schif","schmo","schul","schwa","scifi","scind","scire","sclim","scobe","scody","scogs","scoog","scoot","scopa","scops","scorp","scote","scots","scoug","scoup","scowp","scows","scrab","scrae","scrag","scran","scrat","scraw","scray","scrim","scrip","scrob","scrod","scrog","scroo","scrow","scudi","scudo","scuds","scuff","scuft","scugs","sculk","scull","sculp","sculs","scums","scups","scurf","scurs","scuse","scuta","scute","scuts","scuzz","scyes","sdayn","sdein","seals","seame","seams","seamy","seans","seare","sears","sease","seats","seaze","sebum","secco","sechs","sects","seder","sedes","sedge","sedgy","sedum","seeds","seeks","seeld","seels","seely","seems","seeps","seepy","seers","sefer","segar","segas","segni","segno","segol","segos","sehri","seifs","seils","seine","seirs","seise","seism","seity","seiza","sekos","sekts","selah","seles","selfs","selfy","selky","sella","selle","sells","selva","semas","semee","semes","semie","semis","senas","sends","senes","senex","sengi","senna","senor","sensa","sensi","sensu","sente","senti","sents","senvy","senza","sepad","sepal","sepic","sepoy","seppo","septa","septs","serac","serai","seral","sered","serer","seres","serfs","serge","seria","seric","serin","serir","serks","seron","serow","serra","serre","serrs","serry","servo","sesey","sessa","setae","setal","seter","seths","seton","setts","sevak","sevir","sewan","sewar","sewed","sewel","sewen","sewin","sexed","sexer","sexes","sexor","sexto","sexts","seyen","sezes","shads","shags","shahs","shaka","shako","shakt","shalm","shaly","shama","shams","shand","shans","shaps","sharn","shart","shash","shaul","shawm","shawn","shaws","shaya","shays","shchi","sheaf","sheal","sheas","sheds","sheel","shend","sheng","shent","sheol","sherd","shere","shero","shets","sheva","shewn","shews","shiai","shiel","shier","shies","shill","shily","shims","shins","shiok","ships","shirr","shirs","shish","shiso","shist","shite","shits","shiur","shiva","shive","shivs","shlep","shlub","shmek","shmoe","shoat","shoed","shoer","shoes","shogi","shogs","shoji","shojo","shola","shonk","shool","shoon","shoos","shope","shops","shorl","shote","shots","shott","shoud","showd","shows","shoyu","shred","shris","shrow","shtar","shtik","shtum","shtup","shuba","shule","shuln","shuls","shuns","shura","shute","shuts","shwas","shyer","sials","sibbs","sibia","sibyl","sices","sicht","sicko","sicks","sicky","sidas","sided","sider","sides","sidey","sidha","sidhe","sidle","sield","siens","sient","sieth","sieur","sifts","sighs","sigil","sigla","signa","signs","sigri","sijos","sikas","siker","sikes","silds","siled","silen","siler","siles","silex","silks","sills","silos","silts","silty","silva","simar","simas","simba","simis","simps","simul","sinds","sined","sines","sings","sinhs","sinks","sinky","sinsi","sinus","siped","sipes","sippy","sired","siree","sires","sirih","siris","siroc","sirra","sirup","sisal","sises","sista","sists","sitar","sitch","sited","sites","sithe","sitka","situp","situs","siver","sixer","sixes","sixmo","sixte","sizar","sized","sizel","sizer","sizes","skags","skail","skald","skank","skarn","skart","skats","skatt","skaws","skean","skear","skeds","skeed","skeef","skeen","skeer","skees","skeet","skeev","skeez","skegg","skegs","skein","skelf","skell","skelm","skelp","skene","skens","skeos","skeps","skerm","skers","skets","skews","skids","skied","skies","skiey","skimo","skims","skink","skins","skint","skios","skips","skirl","skirr","skite","skits","skive","skivy","sklim","skoal","skobe","skody","skoff","skofs","skogs","skols","skool","skort","skosh","skran","skrik","skroo","skuas","skugs","skyed","skyer","skyey","skyfs","skyre","skyrs","skyte","slabs","slade","slaes","slags","slaid","slake","slams","slane","slank","slaps","slart","slats","slaty","slave","slaws","slays","slebs","sleds","sleer","slews","sleys","slier","slily","slims","slipe","slips","slipt","slish","slits","slive","sloan","slobs","sloes","slogs","sloid","slojd","sloka","slomo","sloom","sloot","slops","slopy","slorm","slots","slove","slows","sloyd","slubb","slubs","slued","slues","sluff","slugs","sluit","slums","slurb","slurs","sluse","sluts","slyer","slype","smaak","smaik","smalm","smalt","smarm","smaze","smeek","smees","smeik","smeke","smerk","smews","smick","smily","smirr","smirs","smits","smize","smogs","smoko","smolt","smoor","smoot","smore","smorg","smout","smowt","smugs","smurs","smush","smuts","snabs","snafu","snags","snaps","snarf","snark","snars","snary","snash","snath","snaws","snead","sneap","snebs","sneck","sneds","sneed","snees","snell","snibs","snick","snied","snies","snift","snigs","snips","snipy","snirt","snits","snive","snobs","snods","snoek","snoep","snogs","snoke","snood","snook","snool","snoot","snots","snowk","snows","snubs","snugs","snush","snyes","soaks","soaps","soare","soars","soave","sobas","socas","soces","socia","socko","socks","socle","sodas","soddy","sodic","sodom","sofar","sofas","softa","softs","softy","soger","sohur","soils","soily","sojas","sojus","sokah","soken","sokes","sokol","solah","solan","solas","solde","soldi","soldo","solds","soled","solei","soler","soles","solon","solos","solum","solus","soman","somas","sonce","sonde","sones","songo","songs","songy","sonly","sonne","sonny","sonse","sonsy","sooey","sooks","sooky","soole","sools","sooms","soops","soote","soots","sophs","sophy","sopor","soppy","sopra","soral","soras","sorbi","sorbo","sorbs","sorda","sordo","sords","sored","soree","sorel","sorer","sores","sorex","sorgo","sorns","sorra","sorta","sorts","sorus","soths","sotol","sotto","souce","souct","sough","souks","souls","souly","soums","soups","soupy","sours","souse","souts","sowar","sowce","sowed","sowff","sowfs","sowle","sowls","sowms","sownd","sowne","sowps","sowse","sowth","soxes","soyas","soyle","soyuz","sozin","spack","spacy","spado","spads","spaed","spaer","spaes","spags","spahi","spail","spain","spait","spake","spald","spale","spall","spalt","spams","spane","spang","spans","spard","spars","spart","spate","spats","spaul","spawl","spaws","spayd","spays","spaza","spazz","speal","spean","speat","specs","spect","speel","speer","speil","speir","speks","speld","spelk","speos","spesh","spets","speug","spews","spewy","spial","spica","spick","spics","spide","spier","spies","spiff","spifs","spiks","spile","spims","spina","spink","spins","spirt","spiry","spits","spitz","spivs","splay","splog","spode","spods","spoom","spoor","spoot","spork","sposa","sposh","sposo","spots","sprad","sprag","sprat","spred","sprew","sprit","sprod","sprog","sprue","sprug","spuds","spued","spuer","spues","spugs","spule","spume","spumy","spurs","sputa","spyal","spyre","squab","squaw","squee","squeg","squid","squit","squiz","srsly","stabs","stade","stags","stagy","staig","stane","stang","stans","staph","staps","starn","starr","stars","stary","stats","statu","staun","staws","stays","stean","stear","stedd","stede","steds","steek","steem","steen","steez","steik","steil","stela","stele","stell","steme","stems","stend","steno","stens","stent","steps","stept","stere","stets","stews","stewy","steys","stich","stied","sties","stilb","stile","stime","stims","stimy","stipa","stipe","stire","stirk","stirp","stirs","stive","stivy","stoae","stoai","stoas","stoat","stobs","stoep","stogs","stogy","stoit","stoln","stoma","stond","stong","stonk","stonn","stook","stoor","stope","stops","stopt","stoss","stots","stott","stoun","stoup","stour","stown","stowp","stows","strad","strae","strag","strak","strep","strew","stria","strig","strim","strop","strow","stroy","strum","stubs","stucs","stude","studs","stull","stulm","stumm","stums","stuns","stupa","stupe","sture","sturt","stush","styed","styes","styli","stylo","styme","stymy","styre","styte","subah","subak","subas","subby","suber","subha","succi","sucks","sucky","sucre","sudan","sudds","sudor","sudsy","suede","suent","suers","suete","suets","suety","sugan","sughs","sugos","suhur","suids","suint","suits","sujee","sukhs","sukis","sukuk","sulci","sulfa","sulfo","sulks","sulls","sulph","sulus","sumis","summa","sumos","sumph","sumps","sunis","sunks","sunna","sunns","sunts","sunup","suona","suped","supes","supra","surah","sural","suras","surat","surds","sured","sures","surfs","surfy","surgy","surra","sused","suses","susus","sutor","sutra","sutta","swabs","swack","swads","swage","swags","swail","swain","swale","swaly","swamy","swang","swank","swans","swaps","swapt","sward","sware","swarf","swart","swats","swayl","sways","sweal","swede","sweed","sweel","sweer","swees","sweir","swelt","swerf","sweys","swies","swigs","swile","swims","swink","swipe","swire","swiss","swith","swits","swive","swizz","swobs","swole","swoll","swoln","swops","swopt","swots","swoun","sybbe","sybil","syboe","sybow","sycee","syces","sycon","syeds","syens","syker","sykes","sylis","sylph","sylva","symar","synch","syncs","synds","syned","synes","synth","syped","sypes","syphs","syrah","syren","sysop","sythe","syver","taals","taata","tabac","taber","tabes","tabid","tabis","tabla","tabls","tabor","tabos","tabun","tabus","tacan","taces","tacet","tache","tachi","tacho","tachs","tacks","tacos","tacts","tadah","taels","tafia","taggy","tagma","tagua","tahas","tahrs","taiga","taigs","taiko","tails","tains","taira","taish","taits","tajes","takas","takes","takhi","takht","takin","takis","takky","talak","talaq","talar","talas","talcs","talcy","talea","taler","tales","talik","talks","talky","talls","talma","talpa","taluk","talus","tamal","tamas","tamed","tames","tamin","tamis","tammy","tamps","tanas","tanga","tangi","tangs","tanhs","tania","tanka","tanks","tanky","tanna","tansu","tansy","tante","tanti","tanto","tanty","tapas","taped","tapen","tapes","tapet","tapis","tappa","tapus","taras","tardo","tards","tared","tares","targa","targe","tarka","tarns","taroc","tarok","taros","tarps","tarre","tarry","tarse","tarsi","tarte","tarts","tarty","tarzy","tasar","tasca","tased","taser","tases","tasks","tassa","tasse","tasso","tasto","tatar","tater","tates","taths","tatie","tatou","tatts","tatus","taube","tauld","tauon","taupe","tauts","tauty","tavah","tavas","taver","tawaf","tawai","tawas","tawed","tawer","tawie","tawse","tawts","taxed","taxer","taxes","taxis","taxol","taxon","taxor","taxus","tayra","tazza","tazze","teade","teads","teaed","teaks","teals","teams","tears","teats","teaze","techs","techy","tecta","tecum","teels","teems","teend","teene","teens","teeny","teers","teets","teffs","teggs","tegua","tegus","tehee","tehrs","teiid","teils","teind","teins","tekke","telae","telco","teles","telex","telia","telic","tells","telly","teloi","telos","temed","temes","tempi","temps","tempt","temse","tench","tends","tendu","tenes","tenge","tenia","tenne","tenno","tenny","tenon","tents","tenty","tenue","tepal","tepas","tepoy","terai","teras","terce","terek","teres","terfe","terfs","terga","terms","terne","terns","terre","terry","terts","terza","tesla","testa","teste","tests","tetes","teths","tetra","tetri","teuch","teugh","tewed","tewel","tewit","texas","texes","texta","texts","thack","thagi","thaim","thale","thali","thana","thane","thang","thans","thanx","tharm","thars","thaws","thawt","thawy","thebe","theca","theed","theek","thees","thegn","theic","thein","thelf","thema","thens","theor","theow","therm","thesp","thete","thews","thewy","thigs","thilk","thill","thine","thins","thiol","thirl","thoft","thole","tholi","thoro","thorp","thots","thous","thowl","thrae","thraw","thrid","thrip","throe","thuds","thugs","thuja","thunk","thurl","thuya","thymi","thymy","tians","tiare","tiars","tical","ticca","ticed","tices","tichy","ticks","ticky","tiddy","tided","tides","tiefs","tiers","tiffs","tifos","tifts","tiges","tigon","tikas","tikes","tikia","tikis","tikka","tilak","tiled","tiler","tiles","tills","tilly","tilth","tilts","timbo","timed","times","timon","timps","tinas","tinct","tinds","tinea","tined","tines","tinge","tings","tinks","tinny","tinto","tints","tinty","tipis","tippy","tipup","tired","tires","tirls","tiros","tirrs","tirth","titar","titas","titch","titer","tithi","titin","titir","titis","titre","titty","titup","tiyin","tiyns","tizes","tizzy","toads","toady","toaze","tocks","tocky","tocos","todde","todea","todos","toeas","toffs","toffy","tofts","tofus","togae","togas","toged","toges","togue","tohos","toidy","toile","toils","toing","toise","toits","toity","tokay","toked","toker","tokes","tokos","tolan","tolar","tolas","toled","toles","tolls","tolly","tolts","tolus","tolyl","toman","tombo","tombs","tomen","tomes","tomia","tomin","tomme","tommy","tomos","tomoz","tondi","tondo","toned","toner","tones","toney","tongs","tonka","tonks","tonne","tonus","tools","tooms","toons","toots","toped","topee","topek","toper","topes","tophe","tophi","tophs","topis","topoi","topos","toppy","toque","torah","toran","toras","torcs","tores","toric","torii","toros","torot","torrs","torse","torsi","torsk","torta","torte","torts","tosas","tosed","toses","toshy","tossy","tosyl","toted","toter","totes","totty","touks","touns","tours","touse","tousy","touts","touze","touzy","towai","towed","towie","towno","towns","towny","towse","towsy","towts","towze","towzy","toyed","toyer","toyon","toyos","tozed","tozes","tozie","trabs","trads","trady","traga","tragi","trags","tragu","traik","trams","trank","tranq","trans","trant","trape","trapo","traps","trapt","trass","trats","tratt","trave","trayf","trays","treck","treed","treen","trees","trefa","treif","treks","trema","trems","tress","trest","trets","trews","treyf","treys","triac","tride","trier","tries","trifa","triff","trigo","trigs","trike","trild","trill","trims","trine","trins","triol","trior","trios","trips","tripy","trist","troad","troak","troat","trock","trode","trods","trogs","trois","troke","tromp","trona","tronc","trone","tronk","trons","trooz","tropo","troth","trots","trows","troys","trued","trues","trugo","trugs","trull","tryer","tryke","tryma","tryps","tsade","tsadi","tsars","tsked","tsuba","tsubo","tuans","tuart","tuath","tubae","tubar","tubas","tubby","tubed","tubes","tucks","tufas","tuffe","tuffs","tufts","tufty","tugra","tuile","tuina","tuism","tuktu","tules","tulpa","tulps","tulsi","tumid","tummy","tumps","tumpy","tunas","tunds","tuned","tuner","tunes","tungs","tunny","tupek","tupik","tuple","tuque","turds","turfs","turfy","turks","turme","turms","turns","turnt","turon","turps","turrs","tushy","tusks","tusky","tutee","tutes","tutti","tutty","tutus","tuxes","tuyer","twaes","twain","twals","twank","twats","tways","tweel","tween","tweep","tweer","twerk","twerp","twier","twigs","twill","twilt","twink","twins","twiny","twire","twirk","twirp","twite","twits","twocs","twoer","twonk","twyer","tyees","tyers","tyiyn","tykes","tyler","tymps","tynde","tyned","tynes","typal","typed","types","typey","typic","typos","typps","typto","tyran","tyred","tyres","tyros","tythe","tzars","ubacs","ubity","udals","udons","udyog","ugali","ugged","uhlan","uhuru","ukase","ulama","ulans","ulema","ulmin","ulmos","ulnad","ulnae","ulnar","ulnas","ulpan","ulvas","ulyie","ulzie","umami","umbel","umber","umble","umbos","umbre","umiac","umiak","umiaq","ummah","ummas","ummed","umped","umphs","umpie","umpty","umrah","umras","unagi","unais","unapt","unarm","unary","unaus","unbag","unban","unbar","unbed","unbid","unbox","uncap","unces","uncia","uncos","uncoy","uncus","undam","undee","undos","undug","uneth","unfix","ungag","unget","ungod","ungot","ungum","unhat","unhip","unica","unios","units","unjam","unked","unket","unkey","unkid","unkut","unlap","unlaw","unlay","unled","unleg","unlet","unlid","unmad","unman","unmew","unmix","unode","unold","unown","unpay","unpeg","unpen","unpin","unply","unpot","unput","unred","unrid","unrig","unrip","unsaw","unsay","unsee","unsew","unsex","unsod","unsub","untag","untax","untin","unwet","unwit","unwon","upbow","upbye","updos","updry","upend","upful","upjet","uplay","upled","uplit","upped","upran","uprun","upsee","upsey","uptak","upter","uptie","uraei","urali","uraos","urare","urari","urase","urate","urbex","urbia","urdee","ureal","ureas","uredo","ureic","ureid","urena","urent","urged","urger","urges","urial","urite","urman","urnal","urned","urped","ursae","ursid","urson","urubu","urupa","urvas","usens","users","useta","usnea","usnic","usque","ustad","uster","usure","usury","uteri","utero","uveal","uveas","uvula","vacas","vacay","vacua","vacui","vacuo","vadas","vaded","vades","vadge","vagal","vagus","vaids","vails","vaire","vairs","vairy","vajra","vakas","vakil","vales","valis","valli","valse","vamps","vampy","vanda","vaned","vanes","vanga","vangs","vants","vaped","vaper","vapes","varan","varas","varda","vardo","vardy","varec","vares","varia","varix","varna","varus","varve","vasal","vases","vasts","vasty","vatas","vatha","vatic","vatje","vatos","vatus","vauch","vaute","vauts","vawte","vaxes","veale","veals","vealy","veena","veeps","veers","veery","vegas","veges","veggo","vegie","vegos","vehme","veils","veily","veins","veiny","velar","velds","veldt","veles","vells","velum","venae","venal","venas","vends","vendu","veney","venge","venin","venti","vents","venus","verba","verbs","verde","verra","verre","verry","versa","verst","verte","verts","vertu","vespa","vesta","vests","vetch","veuve","veves","vexed","vexer","vexes","vexil","vezir","vials","viand","vibed","vibes","vibex","vibey","viced","vices","vichy","vicus","viers","vieux","views","viewy","vifda","viffs","vigas","vigia","vilde","viler","ville","villi","vills","vimen","vinal","vinas","vinca","vined","viner","vines","vinew","vinho","vinic","vinny","vinos","vints","viold","viols","vired","vireo","vires","virga","virge","virgo","virid","virls","virtu","visas","vised","vises","visie","visna","visne","vison","visto","vitae","vitas","vitex","vitro","vitta","vivas","vivat","vivda","viver","vives","vivos","vivre","vizir","vizor","vlast","vleis","vlies","vlogs","voars","vobla","vocab","voces","voddy","vodou","vodun","voema","vogie","voici","voids","voile","voips","volae","volar","voled","voles","volet","volke","volks","volta","volte","volti","volts","volva","volve","vomer","voted","votes","vouge","voulu","vowed","vower","voxel","voxes","vozhd","vraic","vrils","vroom","vrous","vrouw","vrows","vuggs","vuggy","vughs","vughy","vulgo","vulns","vulva","vutty","vygie","waacs","wacke","wacko","wacks","wadas","wadds","waddy","waded","wader","wades","wadge","wadis","wadts","waffs","wafts","waged","wages","wagga","wagyu","wahay","wahey","wahoo","waide","waifs","waift","wails","wains","wairs","waite","waits","wakas","waked","waken","waker","wakes","wakfs","waldo","walds","waled","waler","wales","walie","walis","walks","walla","walls","wally","walty","wamed","wames","wamus","wands","waned","wanes","waney","wangs","wanks","wanky","wanle","wanly","wanna","wanta","wants","wanty","wanze","waqfs","warbs","warby","wards","wared","wares","warez","warks","warms","warns","warps","warre","warst","warts","wases","washi","washy","wasms","wasps","waspy","wasts","watap","watts","wauff","waugh","wauks","waulk","wauls","waurs","waved","waves","wavey","wawas","wawes","wawls","waxed","waxer","waxes","wayed","wazir","wazoo","weald","weals","weamb","weans","wears","webby","weber","wecht","wedel","wedgy","weeds","weeis","weeke","weeks","weels","weems","weens","weeny","weeps","weepy","weest","weete","weets","wefte","wefts","weids","weils","weirs","weise","weize","wekas","welds","welke","welks","welkt","wells","welly","welts","wembs","wench","wends","wenge","wenny","wents","werfs","weros","wersh","wests","wetas","wetly","wexed","wexes","whamo","whams","whang","whaps","whare","whata","whats","whaup","whaur","wheal","whear","wheek","wheen","wheep","wheft","whelk","whelm","whens","whets","whews","wheys","whids","whies","whift","whigs","whilk","whims","whins","whios","whips","whipt","whirr","whirs","whish","whiss","whist","whits","whity","whizz","whomp","whoof","whoot","whops","whore","whorl","whort","whoso","whows","whump","whups","whyda","wicca","wicks","wicky","widdy","wides","wiels","wifed","wifes","wifey","wifie","wifts","wifty","wigan","wigga","wiggy","wikis","wilco","wilds","wiled","wiles","wilga","wilis","wilja","wills","wilts","wimps","winds","wined","wines","winey","winge","wings","wingy","winks","winky","winna","winns","winos","winze","wiped","wiper","wipes","wired","wirer","wires","wirra","wirri","wised","wises","wisha","wisht","wisps","wists","witan","wited","wites","withe","withs","withy","wived","wiver","wives","wizen","wizes","wizzo","woads","woady","woald","wocks","wodge","wodgy","woful","wojus","woker","wokka","wolds","wolfs","wolly","wolve","womas","wombs","womby","womyn","wonga","wongi","wonks","wonky","wonts","woods","wooed","woofs","woofy","woold","wools","woons","woops","woopy","woose","woosh","wootz","words","works","worky","worms","wormy","worts","wowed","wowee","wowse","woxen","wrang","wraps","wrapt","wrast","wrate","wrawl","wrens","wrick","wried","wrier","wries","writs","wroke","wroot","wroth","wryer","wuddy","wudus","wuffs","wulls","wunga","wurst","wuses","wushu","wussy","wuxia","wyled","wyles","wynds","wynns","wyted","wytes","wythe","xebec","xenia","xenic","xenon","xeric","xerox","xerus","xoana","xolos","xrays","xviii","xylan","xylem","xylic","xylol","xylyl","xysti","xysts","yaars","yaass","yabas","yabba","yabby","yacca","yacka","yacks","yadda","yaffs","yager","yages","yagis","yagna","yahoo","yaird","yajna","yakka","yakow","yales","yamen","yampa","yampy","yamun","yandy","yangs","yanks","yapok","yapon","yapps","yappy","yarak","yarco","yards","yarer","yarfa","yarks","yarns","yarra","yarrs","yarta","yarto","yates","yatra","yauds","yauld","yaups","yawed","yawey","yawls","yawns","yawny","yawps","yayas","ybore","yclad","ycled","ycond","ydrad","ydred","yeads","yeahs","yealm","yeans","yeard","years","yecch","yechs","yechy","yedes","yeeds","yeeek","yeesh","yeggs","yelks","yells","yelms","yelps","yelts","yenta","yente","yerba","yerds","yerks","yeses","yesks","yests","yesty","yetis","yetts","yeuch","yeuks","yeuky","yeven","yeves","yewen","yexed","yexes","yfere","yiked","yikes","yills","yince","yipes","yippy","yirds","yirks","yirrs","yirth","yites","yitie","ylems","ylide","ylids","ylike","ylkes","ymolt","ympes","yobbo","yobby","yocks","yodel","yodhs","yodle","yogas","yogee","yoghs","yogic","yogin","yogis","yohah","yohay","yoick","yojan","yokan","yoked","yokeg","yokel","yoker","yokes","yokul","yolks","yolky","yolps","yomim","yomps","yonic","yonis","yonks","yonny","yoofs","yoops","yopos","yoppo","yores","yorga","yorks","yorps","youks","yourn","yours","yourt","youse","yowed","yowes","yowie","yowls","yowsa","yowza","yoyos","yrapt","yrent","yrivd","yrneh","ysame","ytost","yuans","yucas","yucca","yucch","yucko","yucks","yucky","yufts","yugas","yuked","yukes","yukky","yukos","yulan","yules","yummo","yummy","yumps","yupon","yuppy","yurta","yurts","yuzus","zabra","zacks","zaida","zaide","zaidy","zaire","zakat","zamac","zamak","zaman","zambo","zamia","zamis","zanja","zante","zanza","zanze","zappy","zarda","zarfs","zaris","zatis","zawns","zaxes","zayde","zayin","zazen","zeals","zebec","zebub","zebus","zedas","zeera","zeins","zendo","zerda","zerks","zeros","zests","zetas","zexes","zezes","zhomo","zhush","zhuzh","zibet","ziffs","zigan","zikrs","zilas","zilch","zilla","zills","zimbi","zimbs","zinco","zincs","zincy","zineb","zines","zings","zingy","zinke","zinky","zinos","zippo","zippy","ziram","zitis","zitty","zizel","zizit","zlote","zloty","zoaea","zobos","zobus","zocco","zoeae","zoeal","zoeas","zoism","zoist","zokor","zolle","zombi","zonae","zonda","zoned","zoner","zones","zonks","zooea","zooey","zooid","zooks","zooms","zoomy","zoons","zooty","zoppa","zoppo","zoril","zoris","zorro","zorse","zouks","zowee","zowie","zulus","zupan","zupas","zuppa","zurfs","zuzim","zygal","zygon","zymes","zymic","cigar","rebut","sissy","humph","awake","blush","focal","evade","naval","serve","heath","dwarf","model","karma","stink","grade","quiet","bench","abate","feign","major","death","fresh","crust","stool","colon","abase","marry","react","batty","pride","floss","helix","croak","staff","paper","unfed","whelp","trawl","outdo","adobe","crazy","sower","repay","digit","crate","cluck","spike","mimic","pound","maxim","linen","unmet","flesh","booby","forth","first","stand","belly","ivory","seedy","print","yearn","drain","bribe","stout","panel","crass","flume","offal","agree","error","swirl","argue","bleed","delta","flick","totem","wooer","front","shrub","parry","biome","lapel","start","greet","goner","golem","lusty","loopy","round","audit","lying","gamma","labor","islet","civic","forge","corny","moult","basic","salad","agate","spicy","spray","essay","fjord","spend","kebab","guild","aback","motor","alone","hatch","hyper","thumb","dowry","ought","belch","dutch","pilot","tweed","comet","jaunt","enema","steed","abyss","growl","fling","dozen","boozy","erode","world","gouge","click","briar","great","altar","pulpy","blurt","coast","duchy","groin","fixer","group","rogue","badly","smart","pithy","gaudy","chill","heron","vodka","finer","surer","radio","rouge","perch","retch","wrote","clock","tilde","store","prove","bring","solve","cheat","grime","exult","usher","epoch","triad","break","rhino","viral","conic","masse","sonic","vital","trace","using","peach","champ","baton","brake","pluck","craze","gripe","weary","picky","acute","ferry","aside","tapir","troll","unify","rebus","boost","truss","siege","tiger","banal","slump","crank","gorge","query","drink","favor","abbey","tangy","panic","solar","shire","proxy","point","robot","prick","wince","crimp","knoll","sugar","whack","mount","perky","could","wrung","light","those","moist","shard","pleat","aloft","skill","elder","frame","humor","pause","ulcer","ultra","robin","cynic","aroma","caulk","shake","dodge","swill","tacit","other","thorn","trove","bloke","vivid","spill","chant","choke","rupee","nasty","mourn","ahead","brine","cloth","hoard","sweet","month","lapse","watch","today","focus","smelt","tease","cater","movie","saute","allow","renew","their","slosh","purge","chest","depot","epoxy","nymph","found","shall","stove","lowly","snout","trope","fewer","shawl","natal","comma","foray","scare","stair","black","squad","royal","chunk","mince","shame","cheek","ample","flair","foyer","cargo","oxide","plant","olive","inert","askew","heist","shown","zesty","trash","larva","forgo","story","hairy","train","homer","badge","midst","canny","shine","gecko","farce","slung","tipsy","metal","yield","delve","being","scour","glass","gamer","scrap","money","hinge","album","vouch","asset","tiara","crept","bayou","atoll","manor","creak","showy","phase","froth","depth","gloom","flood","trait","girth","piety","goose","float","donor","atone","primo","apron","blown","cacao","loser","input","gloat","awful","brink","smite","beady","rusty","retro","droll","gawky","hutch","pinto","egret","lilac","sever","field","fluff","agape","voice","stead","berth","madam","night","bland","liver","wedge","roomy","wacky","flock","angry","trite","aphid","tryst","midge","power","elope","cinch","motto","stomp","upset","bluff","cramp","quart","coyly","youth","rhyme","buggy","alien","smear","unfit","patty","cling","glean","label","hunky","khaki","poker","gruel","twice","twang","shrug","treat","waste","merit","woven","needy","clown","irony","ruder","gauze","chief","onset","prize","fungi","charm","gully","inter","whoop","taunt","leery","class","theme","lofty","tibia","booze","alpha","thyme","doubt","parer","chute","stick","trice","alike","recap","saint","glory","grate","admit","brisk","soggy","usurp","scald","scorn","leave","twine","sting","bough","marsh","sloth","dandy","vigor","howdy","enjoy","valid","ionic","equal","floor","catch","spade","stein","exist","quirk","denim","grove","spiel","mummy","fault","foggy","flout","carry","sneak","libel","waltz","aptly","piney","inept","aloud","photo","dream","stale","unite","snarl","baker","there","glyph","pooch","hippy","spell","folly","louse","gulch","vault","godly","threw","fleet","grave","inane","shock","crave","spite","valve","skimp","claim","rainy","musty","pique","daddy","quasi","arise","aging","valet","opium","avert","stuck","recut","mulch","genre","plume","rifle","count","incur","total","wrest","mocha","deter","study","lover","safer","rivet","funny","smoke","mound","undue","sedan","pagan","swine","guile","gusty","equip","tough","canoe","chaos","covet","human","udder","lunch","blast","stray","manga","melee","lefty","quick","paste","given","octet","risen","groan","leaky","grind","carve","loose","sadly","spilt","apple","slack","honey","final","sheen","eerie","minty","slick","derby","wharf","spelt","coach","erupt","singe","price","spawn","fairy","jiffy","filmy","stack","chose","sleep","ardor","nanny","niece","woozy","handy","grace","ditto","stank","cream","usual","diode","valor","angle","ninja","muddy","chase","reply","prone","spoil","heart","shade","diner","arson","onion","sleet","dowel","couch","palsy","bowel","smile","evoke","creek","lance","eagle","idiot","siren","built","embed","award","dross","annul","goody","frown","patio","laden","humid","elite","lymph","edify","might","reset","visit","gusto","purse","vapor","crock","write","sunny","loath","chaff","slide","queer","venom","stamp","sorry","still","acorn","aping","pushy","tamer","hater","mania","awoke","brawn","swift","exile","birch","lucky","freer","risky","ghost","plier","lunar","winch","snare","nurse","house","borax","nicer","lurch","exalt","about","savvy","toxin","tunic","pried","inlay","chump","lanky","cress","eater","elude","cycle","kitty","boule","moron","tenet","place","lobby","plush","vigil","index","blink","clung","qualm","croup","clink","juicy","stage","decay","nerve","flier","shaft","crook","clean","china","ridge","vowel","gnome","snuck","icing","spiny","rigor","snail","flown","rabid","prose","thank","poppy","budge","fiber","moldy","dowdy","kneel","track","caddy","quell","dumpy","paler","swore","rebar","scuba","splat","flyer","horny","mason","doing","ozone","amply","molar","ovary","beset","queue","cliff","magic","truce","sport","fritz","edict","twirl","verse","llama","eaten","range","whisk","hovel","rehab","macaw","sigma","spout","verve","sushi","dying","fetid","brain","buddy","thump","scion","candy","chord","basin","march","crowd","arbor","gayly","musky","stain","dally","bless","bravo","stung","title","ruler","kiosk","blond","ennui","layer","fluid","tatty","score","cutie","zebra","barge","matey","bluer","aider","shook","river","privy","betel","frisk","bongo","begun","azure","weave","genie","sound","glove","braid","scope","wryly","rover","assay","ocean","bloom","irate","later","woken","silky","wreck","dwelt","slate","smack","solid","amaze","hazel","wrist","jolly","globe","flint","rouse","civil","vista","relax","cover","alive","beech","jetty","bliss","vocal","often","dolly","eight","joker","since","event","ensue","shunt","diver","poser","worst","sweep","alley","creed","anime","leafy","bosom","dunce","stare","pudgy","waive","choir","stood","spoke","outgo","delay","bilge","ideal","clasp","seize","hotly","laugh","sieve","block","meant","grape","noose","hardy","shied","drawl","daisy","putty","strut","burnt","tulip","crick","idyll","vixen","furor","geeky","cough","naive","shoal","stork","bathe","aunty","check","prime","brass","outer","furry","razor","elect","evict","imply","demur","quota","haven","cavil","swear","crump","dough","gavel","wagon","salon","nudge","harem","pitch","sworn","pupil","excel","stony","cabin","unzip","queen","trout","polyp","earth","storm","until","taper","enter","child","adopt","minor","fatty","husky","brave","filet","slime","glint","tread","steal","regal","guest","every","murky","share","spore","hoist","buxom","inner","otter","dimly","level","sumac","donut","stilt","arena","sheet","scrub","fancy","slimy","pearl","silly","porch","dingo","sepia","amble","shady","bread","friar","reign","dairy","quill","cross","brood","tuber","shear","posit","blank","villa","shank","piggy","freak","which","among","fecal","shell","would","algae","large","rabbi","agony","amuse","bushy","copse","swoon","knife","pouch","ascot","plane","crown","urban","snide","relay","abide","viola","rajah","straw","dilly","crash","amass","third","trick","tutor","woody","blurb","grief","disco","where","sassy","beach","sauna","comic","clued","creep","caste","graze","snuff","frock","gonad","drunk","prong","lurid","steel","halve","buyer","vinyl","utile","smell","adage","worry","tasty","local","trade","finch","ashen","modal","gaunt","clove","enact","adorn","roast","speck","sheik","missy","grunt","snoop","party","touch","mafia","emcee","array","south","vapid","jelly","skulk","angst","tubal","lower","crest","sweat","cyber","adore","tardy","swami","notch","groom","roach","hitch","young","align","ready","frond","strap","puree","realm","venue","swarm","offer","seven","dryer","diary","dryly","drank","acrid","heady","theta","junto","pixie","quoth","bonus","shalt","penne","amend","datum","build","piano","shelf","lodge","suing","rearm","coral","ramen","worth","psalm","infer","overt","mayor","ovoid","glide","usage","poise","randy","chuck","prank","fishy","tooth","ether","drove","idler","swath","stint","while","begat","apply","slang","tarot","radar","credo","aware","canon","shift","timer","bylaw","serum","three","steak","iliac","shirk","blunt","puppy","penal","joist","bunny","shape","beget","wheel","adept","stunt","stole","topaz","chore","fluke","afoot","bloat","bully","dense","caper","sneer","boxer","jumbo","lunge","space","avail","short","slurp","loyal","flirt","pizza","conch","tempo","droop","plate","bible","plunk","afoul","savoy","steep","agile","stake","dwell","knave","beard","arose","motif","smash","broil","glare","shove","baggy","mammy","swamp","along","rugby","wager","quack","squat","snaky","debit","mange","skate","ninth","joust","tramp","spurn","medal","micro","rebel","flank","learn","nadir","maple","comfy","remit","gruff","ester","least","mogul","fetch","cause","oaken","aglow","meaty","gaffe","shyly","racer","prowl","thief","stern","poesy","rocky","tweet","waist","spire","grope","havoc","patsy","truly","forty","deity","uncle","swish","giver","preen","bevel","lemur","draft","slope","annoy","lingo","bleak","ditty","curly","cedar","dirge","grown","horde","drool","shuck","crypt","cumin","stock","gravy","locus","wider","breed","quite","chafe","cache","blimp","deign","fiend","logic","cheap","elide","rigid","false","renal","pence","rowdy","shoot","blaze","envoy","posse","brief","never","abort","mouse","mucky","sulky","fiery","media","trunk","yeast","clear","skunk","scalp","bitty","cider","koala","duvet","segue","creme","super","grill","after","owner","ember","reach","nobly","empty","speed","gipsy","recur","smock","dread","merge","burst","kappa","amity","shaky","hover","carol","snort","synod","faint","haunt","flour","chair","detox","shrew","tense","plied","quark","burly","novel","waxen","stoic","jerky","blitz","beefy","lyric","hussy","towel","quilt","below","bingo","wispy","brash","scone","toast","easel","saucy","value","spice","honor","route","sharp","bawdy","radii","skull","phony","issue","lager","swell","urine","gassy","trial","flora","upper","latch","wight","brick","retry","holly","decal","grass","shack","dogma","mover","defer","sober","optic","crier","vying","nomad","flute","hippo","shark","drier","obese","bugle","tawny","chalk","feast","ruddy","pedal","scarf","cruel","bleat","tidal","slush","semen","windy","dusty","sally","igloo","nerdy","jewel","shone","whale","hymen","abuse","fugue","elbow","crumb","pansy","welsh","syrup","terse","suave","gamut","swung","drake","freed","afire","shirt","grout","oddly","tithe","plaid","dummy","broom","blind","torch","enemy","again","tying","pesky","alter","gazer","noble","ethos","bride","extol","decor","hobby","beast","idiom","utter","these","sixth","alarm","erase","elegy","spunk","piper","scaly","scold","hefty","chick","sooty","canal","whiny","slash","quake","joint","swept","prude","heavy","wield","femme","lasso","maize","shale","screw","spree","smoky","whiff","scent","glade","spent","prism","stoke","riper","orbit","cocoa","guilt","humus","shush","table","smirk","wrong","noisy","alert","shiny","elate","resin","whole","hunch","pixel","polar","hotel","sword","cleat","mango","rumba","puffy","filly","billy","leash","clout","dance","ovate","facet","chili","paint","liner","curio","salty","audio","snake","fable","cloak","navel","spurt","pesto","balmy","flash","unwed","early","churn","weedy","stump","lease","witty","wimpy","spoof","saner","blend","salsa","thick","warty","manic","blare","squib","spoon","probe","crepe","knack","force","debut","order","haste","teeth","agent","widen","icily","slice","ingot","clash","juror","blood","abode","throw","unity","pivot","slept","troop","spare","sewer","parse","morph","cacti","tacky","spool","demon","moody","annex","begin","fuzzy","patch","water","lumpy","admin","omega","limit","tabby","macho","aisle","skiff","basis","plank","verge","botch","crawl","lousy","slain","cubic","raise","wrack","guide","foist","cameo","under","actor","revue","fraud","harpy","scoop","climb","refer","olden","clerk","debar","tally","ethic","cairn","tulle","ghoul","hilly","crude","apart","scale","older","plain","sperm","briny","abbot","rerun","quest","crisp","bound","befit","drawn","suite","itchy","cheer","bagel","guess","broad","axiom","chard","caput","leant","harsh","curse","proud","swing","opine","taste","lupus","gumbo","miner","green","chasm","lipid","topic","armor","brush","crane","mural","abled","habit","bossy","maker","dusky","dizzy","lithe","brook","jazzy","fifty","sense","giant","surly","legal","fatal","flunk","began","prune","small","slant","scoff","torus","ninny","covey","viper","taken","moral","vogue","owing","token","entry","booth","voter","chide","elfin","ebony","neigh","minim","melon","kneed","decoy","voila","ankle","arrow","mushy","tribe","cease","eager","birth","graph","odder","terra","weird","tried","clack","color","rough","weigh","uncut","ladle","strip","craft","minus","dicey","titan","lucid","vicar","dress","ditch","gypsy","pasta","taffy","flame","swoop","aloof","sight","broke","teary","chart","sixty","wordy","sheer","leper","nosey","bulge","savor","clamp","funky","foamy","toxic","brand","plumb","dingy","butte","drill","tripe","bicep","tenor","krill","worse","drama","hyena","think","ratio","cobra","basil","scrum","bused","phone","court","camel","proof","heard","angel","petal","pouty","throb","maybe","fetal","sprig","spine","shout","cadet","macro","dodgy","satyr","rarer","binge","trend","nutty","leapt","amiss","split","myrrh","width","sonar","tower","baron","fever","waver","spark","belie","sloop","expel","smote","baler","above","north","wafer","scant","frill","awash","snack","scowl","frail","drift","limbo","fence","motel","ounce","wreak","revel","talon","prior","knelt","cello","flake","debug","anode","crime","salve","scout","imbue","pinky","stave","vague","chock","fight","video","stone","teach","cleft","frost","prawn","booty","twist","apnea","stiff","plaza","ledge","tweak","board","grant","medic","bacon","cable","brawl","slunk","raspy","forum","drone","women","mucus","boast","toddy","coven","tumor","truer","wrath","stall","steam","axial","purer","daily","trail","niche","mealy","juice","nylon","plump","merry","flail","papal","wheat","berry","cower","erect","brute","leggy","snipe","sinew","skier","penny","jumpy","rally","umbra","scary","modem","gross","avian","greed","satin","tonic","parka","sniff","livid","stark","trump","giddy","reuse","taboo","avoid","quote","devil","liken","gloss","gayer","beret","noise","gland","dealt","sling","rumor","opera","thigh","tonga","flare","wound","white","bulky","etude","horse","circa","paddy","inbox","fizzy","grain","exert","surge","gleam","belle","salvo","crush","fruit","sappy","taker","tract","ovine","spiky","frank","reedy","filth","spasm","heave","mambo","right","clank","trust","lumen","borne","spook","sauce","amber","lathe","carat","corer","dirty","slyly","affix","alloy","taint","sheep","kinky","wooly","mauve","flung","yacht","fried","quail","brunt","grimy","curvy","cagey","rinse","deuce","state","grasp","milky","bison","graft","sandy","baste","flask","hedge","girly","swash","boney","coupe","endow","abhor","welch","blade","tight","geese","miser","mirth","cloud","cabal","leech","close","tenth","pecan","droit","grail","clone","guise","ralph","tango","biddy","smith","mower","payee","serif","drape","fifth","spank","glaze","allot","truck","kayak","virus","testy","tepee","fully","zonal","metro","curry","grand","banjo","axion","bezel","occur","chain","nasal","gooey","filer","brace","allay","pubic","raven","plead","gnash","flaky","munch","dully","eking","thing","slink","hurry","theft","shorn","pygmy","ranch","wring","lemon","shore","mamma","froze","newer","style","moose","antic","drown","vegan","chess","guppy","union","lever","lorry","image","cabby","druid","exact","truth","dopey","spear","cried","chime","crony","stunk","timid","batch","gauge","rotor","crack","curve","latte","witch","bunch","repel","anvil","soapy","meter","broth","madly","dried","scene","known","magma","roost","woman","thong","punch","pasty","downy","knead","whirl","rapid","clang","anger","drive","goofy","email","music","stuff","bleep","rider","mecca","folio","setup","verso","quash","fauna","gummy","happy","newly","fussy","relic","guava","ratty","fudge","femur","chirp","forte","alibi","whine","petty","golly","plait","fleck","felon","gourd","brown","thrum","ficus","stash","decry","wiser","junta","visor","daunt","scree","impel","await","press","whose","turbo","stoop","speak","mangy","eying","inlet","crone","pulse","mossy","staid","hence","pinch","teddy","sully","snore","ripen","snowy","attic","going","leach","mouth","hound","clump","tonal","bigot","peril","piece","blame","haute","spied","undid","intro","basal","rodeo","guard","steer","loamy","scamp","scram","manly","hello","vaunt","organ","feral","knock","extra","condo","adapt","willy","polka","rayon","skirt","faith","torso","match","mercy","tepid","sleek","riser","twixt","peace","flush","catty","login","eject","roger","rival","untie","refit","aorta","adult","judge","rower","artsy","rural","shave","bobby","eclat","fella","gaily","harry","hasty","hydro","liege","octal","ombre","payer","sooth","unset","unlit","vomit","fanny","fetus","butch","stalk","flack","widow","augur"]

# ╔═╡ fa02baf4-dc36-461a-a7ea-0ee22fb6011f
#=╠═╡
@htl("""
Example Guess: $(@bind example_guess Select(nyt_valid_words; default = "apple"))
Example Answer: $(@bind example_answer Select(nyt_valid_words; default = "cloud"))
""")
  ╠═╡ =#

# ╔═╡ 12d7468b-2675-476a-88ce-284e85b4a589
#index mapping valid guesses to a numerical index
const word_index = Dict(zip(nyt_valid_words, eachindex(nyt_valid_words)))

# ╔═╡ d3d33c45-434b-44ff-b4d7-7ba6e1f415fd
guess_random_word() = rand(nyt_valid_words)

# ╔═╡ a3121433-4d2e-4224-905b-aa0f8d91db05
#=╠═╡
md"""
### Precomputed Feedback Values and Distributions
"""
  ╠═╡ =#

# ╔═╡ 82259951-eae3-4f20-89ea-b84144115028
#=╠═╡
md"""
### Visualization of the upper left 100 elements of the feedback matrix
"""
  ╠═╡ =#

# ╔═╡ 8bb972da-20e5-4d0c-b964-b56ba62e631e
#bit vector representing all of the indices for valid guesses
const nyt_valid_inds = BitVector(fill(1, length(nyt_valid_words)))

# ╔═╡ 2b7adbb7-5c64-42fe-8178-3d1187f4b3fb
#bit vector representing all of the answers in the original wordle game
const wordle_original_inds = BitVector(in(nyt_valid_words[i], wordle_original_answers) for i in eachindex(nyt_valid_words))

# ╔═╡ e5b1d8e5-f224-44e3-8190-b8146ed3ea92
#=╠═╡
md"""
### Game Scoring Methodology Based on Information Gain

One way to think about the game is in terms of how many bits of information we gain for each guess.  In this context, information gain is the decrease in entropy of the distribution of what we believe are the remaining answers.  If we have $p_i = \Pr\{\text{guess}_i = \text{answer}\}$ for every guess, then that probability distribution can be used to compute the current entropy: $\text{entropy} = - \sum_i p_i \log p_i$.  For simplicity let's take our prior assumption of the answer distribution to just be a uniform distribution over the original answer words.  For the original game, the number of possible answers was $(length(wordle_original_answers)).  In the case of the uniform distribution $$p_i = \frac{1}{n}$$ where n is the number of non-zero items in the distribution.  So $$\text{entropy}_n = \log{n}$$ and in the case of the wordle original answers, this value is $(log2(length(wordle_original_answers))) bits of information when using the base 2 log.

After each guess, the provided information updates the distribution of possible answers by removing some from the original list, so on each step $n$ will either stay the same (in the case of no new information) or shrink.  When $n=1$, we know the answer and have gained all possible information from the distribution.  The possible answer distribution when we know the answer has the lowest possible entropy of zero.  In this case, we have gained the maximum amount of information possible which is all of the bits from the original distribution.

In the above function definitions are methods to compute the possible answer distribution after a series of guess/feedback pairs.  At any point in the game, this distribution can be computed and used to determine how much information has been gained up to this point and how much remains.  If a Wordle game could extend to an unlimited number of guesses, then we could simply score the game by dividing the information gain by the number of guesses required to win.  Since every game is eventually a win, the amount of information gained is a constant so this is equivalent to scoring shorter games higher and always favoring a faster finish.  In practice, it may make the most sense to allow lost games to continue until a hypothetical win to score policies, but this has the problem of potentially lasting forever.  One variation of the game called *hard mode* requires that guesses are words that still could be possible answers.  Forcing a policy to chose these words would eliminate the scenario where a game lasts forever.  Also, on the sixth guess of a game, if the guess is not the answer then the game is automatically lost, so a hard mode guess should always be forced on the sixth guess regardless of the policy preference.  So one possible approach to scoring games would be to force *hard mode* guesses on the sixth guess and on until the game is won.  By using this approach, lost games which are closer to an answer will be scored higher on average than lost games with many possible answers remaining and the score values are always based on the number of turns played.

Given that we are trying to end the game quickly, how should we make guesses in this very large action space?  One approach is to make guesses with the highest expected information gain based on the current distribution of possible answers.  The expected information gain can be quickly computed for every guess and used to rank guesses from best to worst.  There is one wrinkle with this scoring system though due to the distinction between a win and a loss.  An incorrect guess could narrow down the possible answer pool to 1 and thus provide the maximum possible information gain.  Another guess could do the same but also be the answer.  In the former case, an additional guess would be required in order to win while in the latter case we have won in one fewer turn.  One way to distinguish the cases would be to consider the information gain per guess.  For the cases where we guess the answer, we know just one additional guess was required to win the game.  In the case of an incorrect guess, we are left with a game with one or more possible answers.  If the number of possible answers is just 1, then the information gain will be the same, but the required guesses will be 2.  For all other cases, the number of guesses needed to win will be at least 1 more.  The information gain is smaller already as a starting point, but we do not know for certain how many guesses will be required to win.  If we assume on average just two more are needed then we would divide by 3.  For every scenario there will be a distribution of possible outcomes, so the decision on how badly to punish guesses that don't reveal all of the information will affect the ranking.  
"""
  ╠═╡ =#

# ╔═╡ 32c30d8e-de5b-43a0-bfdb-8fb86037de7f
const wordle_original_entropy = log2(length(wordle_original_answers))

# ╔═╡ be061e94-d403-453a-99fb-7a1e13bebf52
#=╠═╡
md"""
To calculate the possible remaining answers, we must use the feedback information received from each guess.  A game state should contain this information in the form of a list of guesses and the corresponding feedback.
"""
  ╠═╡ =#

# ╔═╡ 783d068c-77b0-43e1-907b-e532317c5afd
import Base.:(==)

# ╔═╡ bb5ef5ff-93fe-4985-a920-442862e4498b
import Base.hash

# ╔═╡ b9a5d547-79b4-4c7c-a3eb-ed7e87513a88
#=╠═╡
md"""
### Wordle Game State Definitions

To define a Wordle state I only need to know the list of guesses and corresponding feedback seen so far.  To save space and ensure that the two match I will use static vectors of the same length and only 8 bits per element for the feedback since the maximum value is 242.  For the guess list, I can use 16 bit unsigned integers because there are only about 15000 possible words.
"""
  ╠═╡ =#

# ╔═╡ 1d5ba870-3110-4576-a116-a8d0a4d84edc
begin
	const wordle_actions = UInt16.(collect(eachindex(nyt_valid_words)))

	struct WordleState{N}
		guess_list::SVector{N, UInt16}
		feedback_list::SVector{N, UInt8}
	end

	#initialize a game start
	WordleState() = WordleState(SVector{0, UInt16}(), SVector{0, UInt8}())

	WordleState(guess_list::AbstractVector{G}, feedback_list::AbstractVector{F}) where {G, F} = WordleState(conv_guess(guess_list), conv_feedback(feedback_list))

	conv_guess(guess::UInt16) = guess
	conv_guess(guess::AbstractString) = word_index[guess]
	conv_guess(guess::Integer) = UInt16(guess)
	conv_guess(guess_list::SVector{N, UInt16}) where N = guess_list
	conv_guess(guess_list::AbstractVector{G}) where {G<:Integer} = SVector{length(guess_list)}(UInt16.(guess_list))
	conv_guess(guess_list::AbstractVector{G}) where {G<:AbstractString} = conv_guess([word_index[w] for w in guess_list])

	conv_feedback(feedback_list::SVector{N, UInt8}) where N = feedback_list
	conv_feedback(feedback_list::AbstractVector{F}) where {F<:Integer} = SVector{length(feedback_list)}(UInt8.(feedback_list))
	conv_feedback(feedback_list::AbstractVector{F}) where {F<:AbstractVector} = conv_feedback([convert_bytes(f) for f in feedback_list])
	
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

# ╔═╡ 16754304-c592-44f3-baec-94a4d824eb49
struct InformationGainScores
	scores::Dict{WordleState, @NamedTuple{guess_scores::Vector{Float32}, guess_entropies::Vector{Float32}, best_guess::UInt16, max_score::Float32, min_score::Float32, access_time::Vector{Float64}}}
	sorted_states::Vector{WordleState}
	sorted_access_times::Vector{Float64}
end

# ╔═╡ 5eeed71e-4171-4063-89ee-90cfa5934413
const information_gain_scores = InformationGainScores(Dict{WordleState, @NamedTuple{guess_scores::Vector{Float32}, guess_entropies::Vector{Float32}, best_guess::UInt16, max_score::Float32, min_score::Float32, access_time::Vector{Float64}}}(), Vector{WordleState}(), Vector{Float64}())

# ╔═╡ 93b857e5-72db-4aaf-abb0-295beab4073c
Base.hash(s::WordleState) = hash(s.guess_list) + hash(s.feedback_list) 

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

# ╔═╡ 852dff48-8c1d-42f2-925e-53f889b1ac6a
begin
	WordleState()
	const greedy_information_gain_lookup = if isfile("greedy_information_gain_lookup.jld2")
		load("greedy_information_gain_lookup.jld2")["greedy_information_gain_lookup"]
	else
		Dict{WordleState, @NamedTuple{best_guess::Int64, best_score::Float32, best_guess_entropy::Float32}}()
	end
	const all_guesses_information_gain_lookup = if isfile("all_guesses_information_gain_lookup.jld2")
		load("all_guesses_information_gain_lookup.jld2")["all_guesses_information_gain_lookup"]
	else
		Dict{WordleState, @NamedTuple{guess_scores::Vector{Float32}, expected_entropy::Vector{Float32}, ranked_guess_inds::Vector{Int64}}}()
	end
end

# ╔═╡ 20a41989-6650-41dd-b34e-fa23c993e669
const state_entropy_lookup = Dict{WordleState, Float32}()

# ╔═╡ 97d40991-d719-4e87-bd70-94551a645448
function display_one_step_guesses(output)
	[(word = nyt_valid_words[i], score = output.guess_scores[i], expected_entropy = output.expected_entropy[i]) for i in output.ranked_guess_inds]
end

# ╔═╡ 92cb4e96-714d-425d-a64b-eff26a5f92ef
#=╠═╡
md"""
Using this one step lookahead method to rank the guesses, we can already come up with a candidate policy to test, one that is greedy with respect to this score.
"""
  ╠═╡ =#

# ╔═╡ 67b9207a-5004-4488-a8e5-43465adbc26b
#=╠═╡
md"""
The following table shows the candidate guesses ranked according to this scoring heuristic in which case the greedy policy will choose the guess with the highest score.  The scores shown are from the state represented below.
"""
  ╠═╡ =#

# ╔═╡ f2ab3716-19a2-4cbb-b46f-a13c297e086d
#=╠═╡
md"""
### Wordle Game Dynamics

In order to simulate games we must have a way of producing the appropriate feedback and taking a game to completion.  By using this type of afterstate MDP all of the transitions are actually deterministic since I have the full probability distribution for every possible outcome
"""
  ╠═╡ =#

# ╔═╡ 66105fd7-aa54-4007-a346-67f0ac7e1188
# show distribution of winning turns for a given guess rather than just mean

# ╔═╡ b1e3f7ea-e2de-4467-b301-0c3cf225e433
const wordle_transition_lookup = Dict{Tuple{WordleState, UInt16}, @NamedTuple{rewards::Vector{Float32}, transition_states::Vector{WordleState}, probabilities::Vector{Float32}}}()

# ╔═╡ 8e2477a9-ca14-4ecf-9194-9bac5ef25fd4
#=╠═╡
md"""
### Visualize Wordle Transition
"""
  ╠═╡ =#

# ╔═╡ 3c2fb945-0173-4de5-9fdf-ea8b9da3cbd3
# list word patterns and show bar to right scaled to probability

# ╔═╡ 961593ef-964c-4363-b4ba-0d45cfe3d198
const test_possible_indices = copy(nyt_valid_inds)

# ╔═╡ b43a3b03-2cb4-4bd7-8476-d5b1480adc0b
const greedy_information_gain_action_lookup = Dict{WordleState, UInt16}()

# ╔═╡ 16f1c036-90c1-4784-b750-3293255e31f7
game_length(::WordleState{N}) where N = N

# ╔═╡ 1eb7de02-ab3c-4c96-82f5-a09c163faa30
function display_word_group(turns::Integer, list::Vector{String})
	l = length(list)
	"""
	<div style = "display: flex;">
	<div style = "width: 7em;">$turns Turns</div>
	<div style = "width: 8em;">$l Answers</div>
	<div style = "width: 50em;">$(isempty(list) ? "" : reduce((a, b) -> "$a, $b", list))</div>
	</div>
	"""
end

# ╔═╡ 50b2a4d0-3f0e-49c6-bc8d-b490b2542bb0
const guess_value_lookup = Dict{Tuple{WordleState, Int64}, Float32}()

# ╔═╡ aaefe619-c7bb-4c1e-acfb-1b2f05ef388d
score2turns(score) = 7f0 - score

# ╔═╡ 9df47c03-18c5-4e22-a6f9-a55ab5e35c39
function make_one_answer_hard_mode(guess_index)
	output = BitVector(fill(false, length(nyt_valid_inds)))
	output[guess_index] = true
	return output
end

# ╔═╡ 4578c627-127e-4a18-972c-03b38cff371b
#=╠═╡
md"""
#### Guess Values for Greedy Information Policy from Starting State
"""
  ╠═╡ =#

# ╔═╡ 3b32e495-98ca-472d-a928-c76fb9302a57
#=╠═╡
md"""
### Wordle Evaluation Test State
"""
  ╠═╡ =#

# ╔═╡ 7f1f2057-24ca-44c0-8496-b6ef68d895bd
#=╠═╡
md"""
### Defining Wordle Distribution MCTS
"""
  ╠═╡ =#

# ╔═╡ 4de12ef4-4e40-41da-aede-e72f8206f173
make_information_gain_prior_args() = (entropies = zeros(Float32, length(nyt_valid_inds)), answer_inds = copy(nyt_valid_inds), possible_answers = copy(wordle_actions), feedback_entropies = zeros(Float32, 243))

# ╔═╡ b61dfb6f-b2d8-48e1-99b4-345a05274dc5
#=╠═╡
md"""
#### Transition States
"""
  ╠═╡ =#

# ╔═╡ 4130d798-e202-4280-b876-9ca989a45a58
HTML("""
<div style = "display: flex;">
<div style = "width: 4em;">Rank</div>
<div style = "width: 10em;">Transition State</div>
<div style = "width: 6em;">Remaining Answers</div>
<div style = "width: 6em;">Simulation Visits</div>
<div style = "width: 6em;">Tree Policy Expected Turns</div>
<div style = "width: 6em;">Greedy Policy Expected Turns</div>
<div style = "width: 6em;">Minimum Possible Turns</div>
<div style = "width: 7em;">Possible Improvement</div>
<div style = "width: 7em;">Tree Improvement</div>
<div>Percent Chance</div>
</div>
""")

# ╔═╡ a7bca65e-e932-4bee-aa4a-bd6da2215472
# ╠═╡ disabled = true
#=╠═╡
explore_tree_state(explore_state, root_wordle_visit_counts, root_wordle_values)
  ╠═╡ =#

# ╔═╡ cdec4e37-6cfc-40c3-9fcc-e7bcb7cbe6e0
function display_policy_compare(d::Dict)
	valid_keys = filter(k -> !isempty(d[k]), keys(d)) |> collect
	improvement_dict = Dict(i => Vector{String}() for i in -6:6)
	for k in valid_keys
		improvement = k[1] - k[2]
		number_of_words = length(d[k])
		for word in d[k]
			push!(improvement_dict[improvement], word) 
		end
	end
	valid_keys = filter(k -> !isempty(improvement_dict[k]), keys(improvement_dict))
	map(collect(valid_keys)) do k
		(policy2_improvement = k, number_of_words = length(improvement_dict[k]), word_list = improvement_dict[k])
	end |> DataFrame
end	

# ╔═╡ 06138a4e-67a3-47a3-84a5-92013fd404ca
# ╠═╡ disabled = true
#=╠═╡
begin
	root_mcts_params
	const base_mcts_options = show_afterstate_mcts_guesses(tree_values, WordleState())
	const mcts_options, set_mcts_options = @use_state(base_mcts_options)
end
  ╠═╡ =#

# ╔═╡ 10d1f403-34c8-46ce-8cfc-d289608f465c
# ╠═╡ disabled = true
#=╠═╡
const track_run, set_run = @use_state("Completed")
  ╠═╡ =#

# ╔═╡ 41e0cddb-e78d-477b-849a-124754340a3c
# ╠═╡ disabled = true
#=╠═╡
if mcts_counter > 0
	@use_effect([]) do
		set_run("Starting mcts evaluation")
		t = time()
		schedule(Task() do
			nsims = 10
			nruns = ceil(Int64, root_mcts_params.nsims / nsims)
			for i in 1:nruns
				stop_mcts_eval > 0 && break	
				elapsed_minutes = (time() - t)/60
				etr = (elapsed_minutes * nruns / i) - elapsed_minutes
				set_run("Running $i of $nruns after $(round(Int64, (time() - t)/60)) minutes.  Estimated $(round(Int64, etr)) minutes left")
				output = @spawn show_afterstate_mcts_guesses(run_wordle_afterstate_mcts(WordleState(), nsims; tree_values = tree_values, sim_message = false, p_scale = 100f0, topn = root_mcts_params.topn), WordleState())
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
else
	set_run("Waiting to run mcts evaluation")
	sleep(0.01)
end
  ╠═╡ =#

# ╔═╡ 03bbb910-dfa7-4c28-b811-afa9e5ca0e63
# ╠═╡ disabled = true
#=╠═╡
track_run
  ╠═╡ =#

# ╔═╡ cb3b46cf-8375-43b2-8736-97882e9b5e18
# ╠═╡ disabled = true
#=╠═╡
display_tree_results(WordleState(), mcts_options)
  ╠═╡ =#

# ╔═╡ 1ba2fd82-f651-43b3-9411-753eef787b68
#idea to show all of the game states or the branching for the base policy from the root state or any later state.  Want to show all of the resulting games and their probability weight or maybe pieces of the tree how it branches down

# ╔═╡ 0812f3c2-35ab-4e2d-87c0-35a7b44af6d4
#another idea is to identify how many and which actions differ from the base policy using a particular starting guess so it would show a state, the policy action in that state, and the tree action instead 

# ╔═╡ d2052e0c-c506-45b5-8deb-76d5e60d300e
const root_guess_candidates = ["trace"] #, "crate", "crane", "slate"]

# ╔═╡ 9ee34b0a-2e11-403f-839b-4e9991bc0eac
const root_guess_bit_filter = BitVector([in(w, root_guess_candidates) for w in nyt_valid_words])

# ╔═╡ 7d4f39fc-0669-4ded-a890-780d3c6b8e70
const root_candidate_tree_fname = "wordle_root_candidate_tree.jld2"

# ╔═╡ b6b40be8-90e3-4335-93ef-f8d92ef1676d
# ╠═╡ disabled = true
#=╠═╡
if isfile(root_candidate_tree_fname)
	mcts_output
	wordle_root_candidate_greedy_information_gain_prior!
	const root_candidate_load = load(root_candidate_tree_fname)
	const root_guess_candidate_visits = root_candidate_load["root_guess_candidate_visits"]
	const root_guess_candidate_values = root_candidate_load["root_guess_candidate_values"]
else
	const root_guess_candidate_visits = deepcopy(mcts_output[2])
	const root_guess_candidate_values = deepcopy(mcts_output[3])
end
  ╠═╡ =#

# ╔═╡ 0da02107-ad39-4c43-9dfc-2e68736d8063
#for root state only select one of the root guess candidates
function wordle_root_candidate_greedy_information_gain_prior!(prior::Vector{Float32}, s::WordleState{0}; kwargs...)
	prior .= root_guess_bit_filter ./ length(root_guess_candidates)
	return word_index[first(root_guess_candidates)]
end

# ╔═╡ 8a9ed8ec-10b6-43e8-bb9a-e8eba96f3ea0
#=╠═╡
if run_root_guess_candidate_mcts > 0
	monte_carlo_tree_search(wordle_mdp, 1f0, WordleState(), wordle_root_candidate_greedy_information_gain_prior!, 100f0, root_guess_candidate_mcts_params.topn;
		nsims = root_guess_candidate_mcts_params.nsims,
		sim_message = true,
		visit_counts = root_guess_candidate_visits,
		Q = root_guess_candidate_values,
		make_step_kwargs = k -> (possible_indices = test_possible_indices,)
		)
	show_wordle_mcts_guesses(root_guess_candidate_visits, root_guess_candidate_values, WordleState())
else
	md"""
	Showing preliminary results for 1 run.  Waiting to run MCTS for $(root_guess_candidate_mcts_params.nsims) simulations
	
	$(show_wordle_mcts_guesses(root_guess_candidate_visits, root_guess_candidate_values, WordleState()))
	"""
end
  ╠═╡ =#

# ╔═╡ a6eb37ac-09fc-4f24-b987-0f1e200dfebc
function create_save_item(d::Dict{S, V}, s::WordleState) where {S<:WordleState, V}
	((s.guess_list, s.feedback_list), d[s])
end

# ╔═╡ 36fb4201-8261-4551-ae38-eba073e3046b
#=╠═╡
const root_candidate_mcts_options, set_root_candidate_mcts_options = @use_state(nothing)
  ╠═╡ =#

# ╔═╡ b3a7619f-82ac-4a6b-9206-ed1b5cfa0078
#=╠═╡
const track_root_candidate_run, set_root_candidate_run = @use_state("Completed")
  ╠═╡ =#

# ╔═╡ 1642481e-8da9-475c-98b4-92e36d90065b
#=╠═╡
track_root_candidate_run
  ╠═╡ =#

# ╔═╡ b9a41bbb-8706-4f1d-beb7-f55ad8ad5ce7
begin
	get_wordle_substate(s::WordleState{0}, N::Integer) = s
	function get_wordle_substate(s::WordleState{N}, N2::Integer) where N
		newN = min(N, N2)
		WordleState(s.guess_list[1:newN], s.feedback_list[1:newN])
	end
end

# ╔═╡ 8327c794-200f-400f-8bdd-d043c548522c
#comparing the tree to the greedy policy can always look at a given state and highlight an action with the most percentage improvement so far and you can rank these improvements down the tree and show a list of ways in which the tree performance is better than the greedy policy.  It would e a list like, in this state you do this instead of what the greedy policy would do and the overall improvement is this, you can show the total tree improvement as well as just the improvement as measured by the greedy policy value.

# ╔═╡ a107e7ff-56e8-4f31-b061-a7895ea29965
#=╠═╡
md"""
### MCTS Evalaution of Wordle Test State
"""
  ╠═╡ =#

# ╔═╡ 23bb63d2-4287-40b8-af9b-89cb98185f17
#for a given feedback, show how likely that was for the guess and how good/bad that is for the policy vs the expected value

# ╔═╡ b9e9f12d-aa40-49d2-9dc2-ac6110d869d7
HTML("""
<div style = "display: flex;">
<div style = "width: 4em;">Rank</div>
<div style = "width: 10em;">Transition State</div>
<div style = "width: 6em;">Remaining Answers</div>
<div style = "width: 6em;">Simulation Visits</div>
<div style = "width: 6em;">Tree Policy Expected Turns</div>
<div style = "width: 6em;">Greedy Policy Expected Turns</div>
<div style = "width: 6em;">Minimum Possible Turns</div>
<div style = "width: 7em;">Possible Improvement</div>
<div style = "width: 7em;">Tree Improvement</div>
<div>Percent Chance</div>
</div>
""")

# ╔═╡ 1e4b84d7-6fe9-4b1e-9f14-3a663421cb1f
Base.string(s::WordleState{0}) = "WordleStart"

# ╔═╡ d4378e47-e231-495c-8e72-de864097421e
Base.show(s::WordleState{0}) = "WordleStart"

# ╔═╡ 13044009-df94-4dd1-93b2-a774015ab1de
Base.display(s::WordleState{0}) = "WordleStart"

# ╔═╡ 9711392c-33e0-4281-a871-7216dc146de6
#=╠═╡
md"""
## Wordle Hard Mode

In hard mode, one is only permitted to make guesses that are in the list of possible remaining answers.  They do not however need to be answers that are known valid answers but any valid guess that is consistent with the information revealed so far.
"""
  ╠═╡ =#

# ╔═╡ 6cb17eea-2ec9-48d5-9900-635687d5300f
const hardmode_scores = Dict{WordleState, @NamedTuple{guess_scores::SparseVector{Float32, Int64}, guess_entropies::SparseVector{Float32, Int64}, best_guess::UInt16, max_score::Float32, min_score::Float32}}()

# ╔═╡ 5d8c0ce9-275b-4a9e-afd3-4c6028b0600c
make_hardmode_information_gain_kwargs() = (scores = zeros(Float32, length(nyt_valid_inds)), entropies = zeros(Float32, length(nyt_valid_inds)), allowed_guess_inds = copy(nyt_valid_inds), possible_answer_inds = copy(nyt_valid_inds), possible_answers = copy(wordle_actions), feedback_entropies = zeros(Float32, 243))

# ╔═╡ 3cb9ba3c-1b9c-4a6c-b8f2-cade651df78d
const hardmode_greedy_lookup = Dict{WordleState, Int64}()

# ╔═╡ d68d26f3-112d-4542-aab8-0477369f3a52
#=╠═╡
md"""
### Wordle Hard Mode MCTS
"""
  ╠═╡ =#

# ╔═╡ f2106242-c6f9-4bb2-85f0-059b3e9db621
#=╠═╡
md"""
## Don't Wordle

In *Don't Wordle* a player must enter six guesses without guessing the true word.  Also all guesses must conform to hard mode rules.  For example, once a letter is yellow, it must appear in all future guesses.  The game is too difficult with only these rules, so a player also has a fixed number of *undos* which remove the previously made guess.  The information revealed by the guess is still available to the player but not counted for determining valid hard mode guesses.  

For Don't Wordle the score can simply be 1 for a win and 0 for a loss since it is not trivial to win every game.  I will start with a version of the game that does not have any undos and then proceed to add undos.  Also, the default heuristic strategy for the game can be minimizing the information gain.
"""
  ╠═╡ =#

# ╔═╡ a0433105-3429-4b03-9f26-b01af2dc3979
const dontwordle_valid_words_raw = "aahed
aalii
aargh
aarti
abaca
abaci
aback
abacs
abaft
abaka
abamp
aband
abase
abash
abask
abate
abaya
abbas
abbed
abbes
abbey
abbot
abcee
abeam
abear
abele
abers
abets
abhor
abide
abies
abled
abler
ables
ablet
ablow
abmho
abode
abohm
aboil
aboma
aboon
abord
abore
abort
about
above
abram
abray
abrim
abrin
abris
absey
absit
abuna
abune
abuse
abuts
abuzz
abyes
abysm
abyss
acais
acari
accas
accoy
acerb
acers
aceta
achar
ached
aches
achoo
acids
acidy
acing
acini
ackee
acker
acmes
acmic
acned
acnes
acock
acold
acorn
acred
acres
acrid
acros
acted
actin
acton
actor
acute
acyls
adage
adapt
adaws
adays
adbot
addax
added
adder
addio
addle
adeem
adept
adhan
adieu
adios
adits
adman
admen
admin
admit
admix
adobe
adobo
adopt
adore
adorn
adown
adoze
adrad
adred
adsum
aduki
adult
adunc
adust
advew
adyta
adzed
adzes
aecia
aedes
aegis
aeons
aerie
aeros
aesir
afald
afara
afars
afear
affix
afire
aflaj
afoot
afore
afoul
afrit
afros
after
again
agama
agami
agape
agars
agast
agate
agave
agaze
agene
agent
agers
agger
aggie
aggri
aggro
aggry
aghas
agila
agile
aging
agios
agism
agist
agita
aglee
aglet
agley
agloo
aglow
aglus
agmas
agoge
agone
agons
agony
agood
agora
agree
agria
agrin
agros
agued
agues
aguna
aguti
ahead
aheap
ahent
ahigh
ahind
ahing
ahint
ahold
ahull
ahuru
aidas
aided
aider
aides
aidoi
aidos
aiery
aigas
aight
ailed
aimed
aimer
ainee
ainga
aioli
aired
airer
airns
airth
airts
aisle
aitch
aitus
aiver
aiyee
aizle
ajies
ajiva
ajuga
ajwan
akees
akela
akene
aking
akita
akkas
alaap
alack
alamo
aland
alane
alang
alans
alant
alapa
alaps
alarm
alary
alate
alays
albas
albee
album
alcid
alcos
aldea
alder
aldol
aleck
alecs
alefs
aleft
aleph
alert
alews
aleye
alfas
algae
algal
algas
algid
algin
algor
algum
alias
alibi
alien
alifs
align
alike
aline
alist
alive
aliya
alkie
alkos
alkyd
alkyl
allay
allee
allel
alley
allis
allod
allot
allow
alloy
allyl
almah
almas
almeh
almes
almud
almug
alods
aloed
aloes
aloft
aloha
aloin
alone
along
aloof
aloos
aloud
alowe
alpha
altar
alter
altho
altos
alula
alums
alure
alvar
alway
amahs
amain
amass
amate
amaut
amaze
amban
amber
ambit
amble
ambos
ambry
ameba
ameer
amend
amene
amens
ament
amias
amice
amici
amide
amido
amids
amies
amiga
amigo
amine
amino
amins
amirs
amiss
amity
amlas
amman
ammon
ammos
amnia
amnic
amnio
amoks
amole
among
amort
amour
amove
amowt
amped
ample
amply
ampul
amrit
amuck
amuse
amyls
anana
anata
ancho
ancle
ancon
andro
anear
anele
anent
angas
angel
anger
angle
anglo
angry
angst
anigh
anile
anils
anima
anime
animi
anion
anise
anker
ankhs
ankle
ankus
anlas
annal
annas
annat
annex
annoy
annul
anoas
anode
anole
anomy
ansae
antae
antar
antas
anted
antes
antic
antis
antra
antre
antsy
anura
anvil
anyon
aorta
apace
apage
apaid
apart
apayd
apays
apeak
apeek
apers
apert
apery
apgar
aphid
aphis
apian
aping
apiol
apish
apism
apnea
apode
apods
apoop
aport
appal
appay
appel
apple
apply
appro
appui
appuy
apres
apron
apses
apsis
apsos
apted
apter
aptly
aquae
aquas
araba
araks
arame
arars
arbas
arbor
arced
archi
arcos
arcus
ardeb
ardor
ardri
aread
areae
areal
arear
areas
areca
aredd
arede
arefy
areic
arena
arene
arepa
arere
arete
arets
arett
argal
argan
argil
argle
argol
argon
argot
argue
argus
arhat
arias
ariel
ariki
arils
ariot
arise
arish
arked
arled
arles
armed
armer
armet
armil
armor
arnas
arnut
aroba
aroha
aroid
aroma
arose
arpas
arpen
arrah
arras
array
arret
arris
arrow
arroz
arsed
arses
arsey
arsis
arson
artal
artel
artic
artis
artsy
aruhe
arums
arval
arvee
arvos
aryls
asana
ascon
ascot
ascus
asdic
ashed
ashen
ashes
ashet
aside
asked
asker
askew
askoi
askos
aspen
asper
aspic
aspie
aspis
aspro
assai
assam
assay
asses
asset
assez
assot
aster
astir
astun
asura
asway
aswim
asyla
ataps
ataxy
atigi
atilt
atimy
atlas
atman
atmas
atmos
atocs
atoke
atoks
atoll
atoms
atomy
atone
atony
atopy
atria
atrip
attap
attar
attic
atuas
audad
audio
audit
auger
aught
augur
aulas
aulic
auloi
aulos
aumil
aunes
aunts
aunty
aurae
aural
aurar
auras
aurei
aures
auric
auris
aurum
autos
auxin
avail
avale
avant
avast
avels
avens
avers
avert
avgas
avian
avine
avion
avise
aviso
avize
avoid
avows
avyze
await
awake
award
aware
awarn
awash
awato
awave
aways
awdls
aweel
aweto
awful
awing
awmry
awned
awner
awoke
awols
awork
axels
axial
axile
axils
axing
axiom
axion
axite
axled
axles
axman
axmen
axoid
axone
axons
ayahs
ayaya
ayelp
aygre
ayins
ayont
ayres
ayrie
azans
azide
azido
azine
azlon
azoic
azole
azons
azote
azoth
azuki
azure
azurn
azury
azygy
azyme
azyms
baaed
baals
babas
babel
babes
babka
baboo
babul
babus
bacca
bacco
baccy
bacha
bachs
backs
bacon
baddy
badge
badly
baels
baffs
baffy
bafts
bagel
baggy
baghs
bagie
bahts
bahus
bahut
bails
bairn
baisa
baith
baits
baiza
baize
bajan
bajra
bajri
bajus
baked
baken
baker
bakes
bakra
balas
balds
baldy
baled
baler
bales
balks
balky
balls
bally
balms
balmy
baloo
balsa
balti
balun
balus
bambi
banak
banal
banco
bancs
banda
bandh
bands
bandy
baned
banes
bangs
bania
banjo
banks
banns
bants
bantu
banty
banya
bapus
barbe
barbs
barby
barca
barde
bardo
bards
bardy
bared
barer
bares
barfi
barfs
barge
baric
barks
barky
barms
barmy
barns
barny
baron
barps
barra
barre
barro
barry
barye
basal
basan
based
basen
baser
bases
basho
basic
basij
basil
basin
basis
basks
bason
basse
bassi
basso
bassy
basta
baste
basti
basto
basts
batch
bated
bates
bathe
baths
batik
baton
batta
batts
battu
batty
bauds
bauks
baulk
baurs
bavin
bawds
bawdy
bawks
bawls
bawns
bawrs
bawty
bayed
bayer
bayes
bayle
bayou
bayts
bazar
bazoo
beach
beads
beady
beaks
beaky
beals
beams
beamy
beano
beans
beany
beard
beare
bears
beast
beath
beats
beaty
beaus
beaut
beaux
bebop
becap
becke
becks
bedad
bedel
bedes
bedew
bedim
bedye
beech
beedi
beefs
beefy
beeps
beers
beery
beets
befit
befog
begad
began
begar
begat
begem
beget
begin
begot
begum
begun
beige
beigy
being
beins
bekah
belah
belar
belay
belch
belee
belga
belie
belle
bells
belly
belon
below
belts
bemad
bemas
bemix
bemud
bench
bends
bendy
benes
benet
benga
benis
benne
benni
benny
bento
bents
benty
bepat
beray
beres
beret
bergs
berko
berks
berme
berms
berob
berry
berth
beryl
besat
besaw
besee
beses
beset
besit
besom
besot
besti
bests
betas
beted
betel
betes
beths
betid
beton
betta
betty
bevel
bever
bevor
bevue
bevvy
bewet
bewig
bezel
bezes
bezil
bezzy
bhais
bhaji
bhang
bhats
bhels
bhoot
bhuna
bhuts
biach
biali
bialy
bibbs
bibes
bible
biccy
bicep
bices
biddy
bided
bider
bides
bidet
bidis
bidon
bield
biers
biffo
biffs
biffy
bifid
bigae
biggs
biggy
bigha
bight
bigly
bigos
bigot
bijou
biked
biker
bikes
bikie
bilbo
bilby
biled
biles
bilge
bilgy
bilks
bills
billy
bimah
bimas
bimbo
binal
bindi
binds
biner
bines
binge
bingo
bings
bingy
binit
binks
bints
biogs
biome
biont
biota
biped
bipod
birch
birds
birks
birle
birls
biros
birrs
birse
birsy
birth
bises
bisks
bisom
bison
bitch
biter
bites
bitos
bitou
bitsy
bitte
bitts
bitty
bivia
bivvy
bizes
bizzo
bizzy
blabs
black
blade
blads
blady
blaer
blaes
blaff
blags
blahs
blain
blame
blams
bland
blank
blare
blart
blase
blash
blast
blate
blats
blatt
blaud
blawn
blaws
blays
blaze
bleak
blear
bleat
blebs
blech
bleed
bleep
blees
blend
blent
blert
bless
blest
blets
bleys
blimp
blimy
blind
bling
blini
blink
blins
bliny
blips
bliss
blist
blite
blits
blitz
blive
bloat
blobs
block
blocs
blogs
bloke
blond
blood
blook
bloom
bloop
blore
blots
blown
blows
blowy
blubs
blude
bluds
bludy
blued
bluer
blues
bluet
bluey
bluff
bluid
blume
blunk
blunt
blurb
blurs
blurt
blush
blype
boabs
boaks
board
boars
boart
boast
boats
bobac
bobak
bobas
bobby
bobol
bobos
bocca
bocce
bocci
boche
bocks
boded
bodes
bodge
bodhi
bodle
boeps
boets
boeuf
boffo
boffs
bogan
bogey
boggy
bogie
bogle
bogue
bogus
bohea
bohos
boils
boing
boink
boite
boked
bokeh
bokes
bokos
bolar
bolas
bolds
boles
bolix
bolls
bolos
bolts
bolus
bomas
bombe
bombo
bombs
bonce
bonds
boned
boner
bones
boney
bongo
bongs
bonie
bonks
bonne
bonny
bonus
bonza
bonze
booai
booay
boobs
booby
boody
booed
boofy
boogy
boohs
books
booky
bools
booms
boomy
boong
boons
boord
boors
boose
boost
booth
boots
booty
booze
boozy
boppy
borak
boral
boras
borax
borde
bords
bored
boree
borel
borer
bores
borgo
boric
borks
borms
borna
borne
boron
borts
borty
bortz
bosie
bosks
bosky
bosom
boson
bossy
bosun
botas
botch
botel
botes
bothy
botte
botts
botty
bouge
bough
bouks
boule
boult
bound
bouns
bourd
bourg
bourn
bouse
bousy
bouts
bovid
bowat
bowed
bowel
bower
bowes
bowet
bowie
bowls
bowne
bowrs
bowse
boxed
boxen
boxer
boxes
boxla
boxty
boyar
boyau
boyed
boyfs
boygs
boyla
boyos
boysy
bozos
braai
brace
brach
brack
bract
brads
braes
brags
braid
brail
brain
brake
braks
braky
brame
brand
brane
brank
brans
brant
brash
brass
brast
brats
brava
brave
bravi
bravo
brawl
brawn
braws
braxy
brays
braza
braze
bread
break
bream
brede
breds
breed
breem
breer
brees
breid
breis
breme
brens
brent
brere
brers
breve
brews
breys
briar
bribe
brick
bride
brief
brier
bries
brigs
briki
briks
brill
brims
brine
bring
brink
brins
briny
brios
brise
brisk
briss
brith
brits
britt
brize
broad
broch
brock
brods
brogh
brogs
broil
broke
brome
bromo
bronc
brond
brood
brook
brool
broom
broos
brose
brosy
broth
brown
brows
brugh
bruin
bruit
brule
brume
brung
brunt
brush
brusk
brust
brute
bruts
buats
buaze
bubal
bubas
bubba
bubbe
bubby
bubus
buchu
bucko
bucks
bucku
budas
buddy
budge
budis
budos
buffa
buffe
buffi
buffo
buffs
buffy
bufos
bufty
buggy
bugle
buhls
buhrs
buiks
build
built
buist
bukes
bulbs
bulge
bulgy
bulks
bulky
bulla
bulls
bully
bulse
bumbo
bumfs
bumph
bumps
bumpy
bunas
bunce
bunch
bunco
bunde
bundh
bunds
bundt
bundu
bundy
bungs
bungy
bunia
bunje
bunjy
bunko
bunks
bunns
bunny
bunts
bunty
bunya
buoys
buppy
buran
buras
burbs
burds
buret
burfi
burgh
burgs
burin
burka
burke
burks
burls
burly
burns
burnt
buroo
burps
burqa
burro
burrs
burry
bursa
burse
burst
busby
bused
buses
bushy
busks
busky
bussu
busti
busts
busty
butch
buteo
butes
butle
butoh
butte
butts
butty
butut
butyl
buxom
buyer
buzzy
bwana
bwazi
byded
bydes
byked
bykes
bylaw
byres
byrls
byssi
bytes
byway
caaed
cabal
cabas
cabby
caber
cabin
cable
cabob
caboc
cabre
cacao
cacas
cache
cacks
cacky
cacti
caddy
cadee
cades
cadet
cadge
cadgy
cadie
cadis
cadre
caeca
caese
cafes
caffs
caged
cager
cages
cagey
cagot
cahow
caids
cains
caird
cairn
cajon
cajun
caked
cakes
cakey
calfs
calid
calif
calix
calks
calla
calls
calms
calmy
calos
calpa
calps
calve
calyx
caman
camas
camel
cameo
cames
camis
camos
campi
campo
camps
campy
camus
canal
candy
caned
caneh
caner
canes
cangs
canid
canna
canns
canny
canoe
canon
canso
canst
canto
cants
canty
capas
caped
caper
capes
capex
caphs
capiz
caple
capon
capos
capot
capri
capul
caput
carap
carat
carbo
carbs
carby
cardi
cards
cardy
cared
carer
cares
caret
carex
cargo
carks
carle
carls
carns
carny
carob
carol
carom
caron
carpi
carps
carrs
carry
carse
carta
carte
carts
carve
carvy
casas
casco
cased
cases
casks
casky
caste
casts
casus
catch
cater
cates
catty
cauda
cauks
cauld
caulk
cauls
caums
caups
cauri
causa
cause
cavas
caved
cavel
caver
caves
cavie
cavil
cawed
cawks
caxon
cease
ceaze
cebid
cecal
cecum
cedar
ceded
ceder
cedes
cedis
ceiba
ceili
ceils
celeb
cella
celli
cello
cells
celom
celts
cense
cento
cents
centu
ceorl
cepes
cerci
cered
ceres
cerge
ceria
ceric
cerne
ceroc
ceros
certs
certy
cesse
cesta
cesti
cetes
cetyl
cezve
chace
chack
chaco
chado
chads
chafe
chaff
chaft
chain
chair
chais
chalk
chals
champ
chams
chana
chang
chank
chant
chaos
chape
chaps
chapt
chara
chard
chare
chark
charm
charr
chars
chart
chary
chase
chasm
chats
chave
chavs
chawk
chaws
chaya
chays
cheap
cheat
check
cheek
cheep
cheer
chefs
cheka
chela
chelp
chemo
chems
chere
chert
chess
chest
cheth
chevy
chews
chewy
chiao
chias
chibs
chica
chich
chick
chico
chics
chide
chief
chiel
chiks
child
chile
chili
chill
chimb
chime
chimo
chimp
china
chine
ching
chink
chino
chins
chips
chirk
chirl
chirm
chiro
chirp
chirr
chirt
chiru
chits
chive
chivs
chivy
chizz
chock
choco
chocs
chode
chogs
choil
choir
choke
choko
choky
chola
choli
cholo
chomp
chons
choof
chook
choom
choon
chops
chord
chore
chose
chota
chott
chout
choux
chowk
chows
chubs
chuck
chufa
chuff
chugs
chump
chums
chunk
churl
churn
churr
chuse
chute
chuts
chyle
chyme
chynd
cibol
cided
cider
cides
ciels
cigar
ciggy
cilia
cills
cimar
cimex
cinch
cinct
cines
cinqs
cions
cippi
circa
circs
cires
cirls
cirri
cisco
cissy
cists
cital
cited
citer
cites
cives
civet
civic
civie
civil
civvy
clach
clack
clade
clads
claes
clags
claim
clame
clamp
clams
clang
clank
clans
claps
clapt
claro
clart
clary
clash
clasp
class
clast
clats
claut
clave
clavi
claws
clays
clean
clear
cleat
cleck
cleek
cleep
clefs
cleft
clegs
cleik
clems
clepe
clept
clerk
cleve
clews
click
clied
clies
cliff
clift
climb
clime
cline
cling
clink
clint
clipe
clips
clipt
clits
cloak
cloam
clock
clods
cloff
clogs
cloke
clomb
clomp
clone
clonk
clons
cloop
cloot
clops
close
clote
cloth
clots
cloud
clour
clous
clout
clove
clown
clows
cloye
cloys
cloze
clubs
cluck
clued
clues
cluey
clump
clung
clunk
clype
cnida
coach
coact
coady
coala
coals
coaly
coapt
coarb
coast
coate
coati
coats
cobbs
cobby
cobia
coble
cobra
cobza
cocas
cocci
cocco
cocks
cocky
cocoa
cocos
codas
codec
coded
coden
coder
codes
codex
codon
coeds
coffs
cogie
cogon
cogue
cohab
cohen
cohoe
cohog
cohos
coifs
coign
coils
coins
coirs
coits
coked
cokes
colas
colby
colds
coled
coles
coley
colic
colin
colls
colly
colog
colon
color
colts
colza
comae
comal
comas
combe
combi
combo
combs
comby
comer
comes
comet
comfy
comic
comix
comma
commo
comms
commy
compo
comps
compt
comte
comus
conch
condo
coned
cones
coney
confs
conga
conge
congo
conia
conic
conin
conks
conky
conne
conns
conte
conto
conus
convo
cooch
cooed
cooee
cooer
cooey
coofs
cooks
cooky
cools
cooly
coomb
cooms
coomy
coons
coops
coopt
coost
coots
cooze
copal
copay
coped
copen
coper
copes
coppy
copra
copse
copsy
coqui
coral
coram
corbe
corby
cords
cored
corer
cores
corey
corgi
coria
corks
corky
corms
corni
corno
corns
cornu
corny
corps
corse
corso
cosec
cosed
coses
coset
cosey
cosie
costa
coste
costs
cotan
coted
cotes
coths
cotta
cotts
couch
coude
cough
could
count
coupe
coups
courb
courd
coure
cours
court
couta
couth
coved
coven
cover
coves
covet
covey
covin
cowal
cowan
cowed
cower
cowks
cowls
cowps
cowry
coxae
coxal
coxed
coxes
coxib
coyau
coyed
coyer
coyly
coypu
cozed
cozen
cozes
cozey
cozie
craal
crabs
crack
craft
crags
craic
craig
crake
crame
cramp
crams
crane
crank
crans
crape
craps
crapy
crare
crash
crass
crate
crave
crawl
craws
crays
craze
crazy
creak
cream
credo
creds
creed
creek
creel
creep
crees
creme
crems
crena
crepe
creps
crept
crepy
cress
crest
crewe
crews
crias
cribs
crick
cried
crier
cries
crime
crimp
crims
crine
crios
cripe
crips
crise
crisp
crith
crits
croak
croci
crock
crocs
croft
crogs
cromb
crome
crone
cronk
crons
crony
crook
crool
croon
crops
crore
cross
crost
croup
crout
crowd
crown
crows
croze
cruck
crude
crudo
cruds
crudy
cruel
crues
cruet
cruft
crumb
crump
crunk
cruor
crura
cruse
crush
crust
crusy
cruve
crwth
cryer
crypt
ctene
cubby
cubeb
cubed
cuber
cubes
cubic
cubit
cuddy
cuffo
cuffs
cuifs
cuing
cuish
cuits
cukes
culch
culet
culex
culls
cully
culms
culpa
culti
cults
culty
cumec
cumin
cundy
cunei
cunit
cunts
cupel
cupid
cuppa
cuppy
curat
curbs
curch
curds
curdy
cured
curer
cures
curet
curfs
curia
curie
curio
curli
curls
curly
curns
curny
currs
curry
curse
cursi
curst
curve
curvy
cusec
cushy
cusks
cusps
cuspy
cusso
cusum
cutch
cuter
cutes
cutey
cutie
cutin
cutis
cutto
cutty
cutup
cuvee
cuzes
cwtch
cyano
cyans
cyber
cycad
cycas
cycle
cyclo
cyder
cylix
cymae
cymar
cymas
cymes
cymol
cynic
cysts
cytes
cyton
czars
daals
dabba
daces
dacha
dacks
dadah
dadas
daddy
dados
daffs
daffy
dagga
daggy
dagos
dahls
daiko
daily
daine
daint
dairy
daisy
daker
daled
dales
dalis
dalle
dally
dalts
daman
damar
dames
damme
damns
damps
dampy
dance
dancy
dandy
dangs
danio
danks
danny
dants
daraf
darbs
darcy
dared
darer
dares
darga
dargs
daric
daris
darks
darky
darns
darre
darts
darzi
dashi
dashy
datal
dated
dater
dates
datos
datto
datum
daube
daubs
dauby
dauds
dault
daunt
daurs
dauts
daven
davit
dawah
dawds
dawed
dawen
dawks
dawns
dawts
dayan
daych
daynt
dazed
dazer
dazes
deads
deair
deals
dealt
deans
deare
dearn
dears
deary
deash
death
deave
deaws
deawy
debag
debar
debby
debel
debes
debit
debts
debud
debug
debur
debus
debut
debye
decad
decaf
decal
decan
decay
decko
decks
decor
decos
decoy
decry
dedal
deeds
deedy
deely
deems
deens
deeps
deere
deers
deets
deeve
deevs
defat
defer
deffo
defis
defog
degas
degum
degus
deice
deids
deify
deign
deils
deism
deist
deity
deked
dekes
dekko
delay
deled
deles
delfs
delft
delis
dells
delly
delos
delph
delta
delts
delve
deman
demes
demic
demit
demob
demoi
demon
demos
dempt
demur
denar
denay
dench
denes
denet
denim
denis
dense
dents
deoxy
depot
depth
derat
deray
derby
dered
deres
derig
derma
derms
derns
derny
deros
derro
derry
derth
dervs
desex
deshi
desis
desks
desse
deter
detox
deuce
devas
devel
devil
devis
devon
devos
devot
dewan
dewar
dewax
dewed
dexes
dexie
dhaba
dhaks
dhals
dhikr
dhobi
dhole
dholl
dhols
dhoti
dhows
dhuti
diact
dials
diane
diary
diazo
dibbs
diced
dicer
dices
dicey
dicht
dicks
dicky
dicot
dicta
dicts
dicty
diddy
didie
didos
didst
diebs
diels
diene
diets
diffs
dight
digit
dikas
diked
diker
dikes
dikey
dildo
dilli
dills
dilly
dimbo
dimer
dimes
dimly
dimps
dinar
dined
diner
dines
dinge
dingo
dings
dingy
dinic
dinks
dinky
dinna
dinos
dints
diode
diols
diota
dippy
dipso
diram
direr
dirge
dirke
dirks
dirls
dirts
dirty
disas
disci
disco
discs
dishy
disks
disme
dital
ditas
ditch
dited
dites
ditsy
ditto
ditts
ditty
ditzy
divan
divas
dived
diver
dives
divis
divna
divos
divot
divvy
diwan
dixie
dixit
diyas
dizen
dizzy
djinn
djins
doabs
doats
dobby
dobes
dobie
dobla
dobra
dobro
docht
docks
docos
docus
doddy
dodge
dodgy
dodos
doeks
doers
doest
doeth
doffs
dogan
doges
dogey
doggo
doggy
dogie
dogma
dohyo
doilt
doily
doing
doits
dojos
dolce
dolci
doled
doles
dolia
dolls
dolly
dolma
dolor
dolos
dolts
domal
domed
domes
domic
donah
donas
donee
doner
donga
dongs
donko
donna
donne
donny
donor
donsy
donut
doobs
dooce
doody
dooks
doole
dools
dooly
dooms
doomy
doona
doorn
doors
doozy
dopas
doped
doper
dopes
dopey
dorad
dorba
dorbs
doree
dores
doric
doris
dorks
dorky
dorms
dormy
dorps
dorrs
dorsa
dorse
dorts
dorty
dosai
dosas
dosed
doseh
doser
doses
dosha
dotal
doted
doter
dotes
dotty
douar
doubt
douce
doucs
dough
douks
doula
douma
doums
doups
doura
douse
douts
doved
doven
dover
doves
dovie
dowar
dowds
dowdy
dowed
dowel
dower
dowie
dowle
dowls
dowly
downa
downs
downy
dowps
dowry
dowse
dowts
doxed
doxes
doxie
doyen
doyly
dozed
dozen
dozer
dozes
drabs
drack
draco
draff
draft
drags
drail
drain
drake
drama
drams
drank
drant
drape
draps
drats
drave
drawl
drawn
draws
drays
dread
dream
drear
dreck
dreed
dreer
drees
dregs
dreks
drent
drere
dress
drest
dreys
dribs
drice
dried
drier
dries
drift
drill
drily
drink
drips
dript
drive
droid
droil
droit
droke
drole
droll
drome
drone
drony
droob
droog
drook
drool
droop
drops
dropt
dross
drouk
drove
drown
drows
drubs
drugs
druid
drums
drunk
drupe
druse
drusy
druxy
dryad
dryas
dryer
dryly
dsobo
dsomo
duads
duals
duans
duars
dubbo
ducal
ducat
duces
duchy
ducks
ducky
ducts
duddy
duded
dudes
duels
duets
duett
duffs
dufus
duing
duits
dukas
duked
dukes
dukka
dulce
dules
dulia
dulls
dully
dulse
dumas
dumbo
dumbs
dumka
dumky
dummy
dumps
dumpy
dunam
dunce
dunch
dunes
dungs
dungy
dunks
dunno
dunny
dunsh
dunts
duomi
duomo
duped
duper
dupes
duple
duply
duppy
dural
duras
dured
dures
durgy
durns
duroc
duros
duroy
durra
durrs
durry
durst
durum
durzi
dusks
dusky
dusts
dusty
dutch
duvet
duxes
dwaal
dwale
dwalm
dwams
dwang
dwarf
dwaum
dweeb
dwell
dwelt
dwile
dwine
dyads
dyers
dying
dyked
dykes
dykey
dykon
dynel
dynes
dzhos
eager
eagle
eagre
ealed
eales
eaned
eards
eared
earls
early
earns
earnt
earst
earth
eased
easel
easer
eases
easle
easts
eaten
eater
eathe
eaved
eaves
ebbed
ebbet
ebons
ebony
ebook
ecads
eched
eches
echos
eclat
ecrus
edema
edged
edger
edges
edict
edify
edile
edits
educe
educt
eejit
eensy
eerie
eeven
eevns
effed
egads
egers
egest
eggar
egged
egger
egmas
egret
ehing
eider
eidos
eight
eigne
eiked
eikon
eilds
eisel
eject
ejido
eking
ekkas
elain
eland
elans
elate
elbow
elchi
elder
eldin
elect
elegy
elemi
elfed
elfin
eliad
elide
elint
elite
elmen
eloge
elogy
eloin
elope
elops
elpee
elsin
elude
elute
elvan
elven
elver
elves
emacs
email
embar
embay
embed
ember
embog
embow
embox
embus
emcee
emeer
emend
emerg
emery
emeus
emics
emirs
emits
emmas
emmer
emmet
emmew
emmys
emoji
emong
emote
emove
empts
empty
emule
emure
emyde
emyds
enact
enarm
enate
ended
ender
endew
endow
endue
enema
enemy
enews
enfix
eniac
enjoy
enlit
enmew
ennog
ennui
enoki
enols
enorm
enows
enrol
ensew
ensky
ensue
enter
entia
entry
enure
enurn
envoi
envoy
enzym
eorls
eosin
epact
epees
ephah
ephas
ephod
ephor
epics
epoch
epode
epopt
epoxy
epris
equal
eques
equid
equip
erase
erbia
erect
erevs
ergon
ergos
ergot
erhus
erica
erick
erics
ering
erned
ernes
erode
erose
erred
error
erses
eruct
erugo
erupt
eruvs
erven
ervil
escar
escot
esile
eskar
esker
esnes
essay
esses
ester
estoc
estop
estro
etage
etape
etats
etens
ethal
ether
ethic
ethne
ethos
ethyl
etics
etnas
ettin
ettle
etude
etuis
etwee
etyma
eughs
euked
eupad
euros
eusol
evade
evens
event
evert
every
evets
evhoe
evict
evils
evite
evohe
evoke
ewers
ewest
ewhow
ewked
exact
exalt
exams
excel
exeat
execs
exeem
exeme
exert
exfil
exies
exile
exine
exing
exist
exits
exode
exome
exons
expat
expel
expos
extol
extra
exude
exuls
exult
exurb
eyass
eyers
eying
eyots
eyras
eyres
eyrie
eyrir
ezine
fabby
fable
faced
facer
faces
facet
facia
facta
facts
faddy
faded
fader
fades
fadge
fados
faena
faery
faffs
faffy
faggy
fagin
fagot
faiks
fails
faine
fains
faint
fairs
fairy
faith
faked
faker
fakes
fakey
fakie
fakir
falaj
falls
false
famed
fames
fanal
fancy
fands
fanes
fanga
fango
fangs
fanks
fanny
fanon
fanos
fanum
faqir
farad
farce
farci
farcy
fards
fared
farer
fares
farle
farls
farms
faros
farro
farse
farts
fasci
fasti
fasts
fatal
fated
fates
fatly
fatso
fatty
fatwa
faugh
fauld
fault
fauna
fauns
faurd
fauts
fauve
favas
favel
faver
faves
favor
favus
fawns
fawny
faxed
faxes
fayed
fayer
fayne
fayre
fazed
fazes
feals
feare
fears
feart
fease
feast
feats
feaze
fecal
feces
fecht
fecit
fecks
fedex
feebs
feeds
feels
feens
feers
feese
feeze
fehme
feign
feint
feist
felch
felid
fella
fells
felly
felon
felts
felty
femal
femes
femme
femmy
femur
fence
fends
fendy
fenis
fenks
fenny
fents
feods
feoff
feral
ferer
feres
feria
ferly
fermi
ferms
ferns
ferny
ferry
fesse
festa
fests
festy
fetal
fetas
fetch
feted
fetes
fetid
fetor
fetta
fetts
fetus
fetwa
feuar
feuds
feued
fever
fewer
feyed
feyer
feyly
fezes
fezzy
fiars
fiats
fiber
fibre
fibro
fices
fiche
fichu
ficin
ficos
ficus
fides
fidge
fidos
fiefs
field
fiend
fient
fiere
fiers
fiery
fiest
fifed
fifer
fifes
fifis
fifth
fifty
figgy
fight
figos
fiked
fikes
filar
filch
filed
filer
files
filet
filii
filks
fille
fillo
fills
filly
filmi
films
filmy
filos
filth
filum
final
finca
finch
finds
fined
finer
fines
finis
finks
finny
finos
fiord
fiqhs
fique
fired
firer
fires
firie
firks
firms
firns
firry
first
firth
fiscs
fishy
fisks
fists
fisty
fitch
fitly
fitna
fitte
fitts
fiver
fives
fixed
fixer
fixes
fixit
fizzy
fjeld
fjord
flabs
flack
flaff
flags
flail
flair
flake
flaks
flaky
flame
flamm
flams
flamy
flane
flank
flans
flaps
flare
flary
flash
flask
flats
flava
flawn
flaws
flawy
flaxy
flays
fleam
fleas
fleck
fleek
fleer
flees
fleet
flegs
fleme
flesh
fleur
flews
flexi
flexo
fleys
flick
flics
flied
flier
flies
flimp
flims
fling
flint
flips
flirs
flirt
flisk
flite
flits
flitt
float
flobs
flock
flocs
floes
flogs
flong
flood
floor
flops
flora
flors
flory
flosh
floss
flota
flote
flour
flout
flown
flows
flubs
flued
flues
fluey
fluff
fluid
fluke
fluky
flume
flump
flung
flunk
fluor
flurr
flush
flute
fluty
fluyt
flyby
flyer
flype
flyte
foals
foams
foamy
focal
focus
foehn
fogey
foggy
fogie
fogle
fogou
fohns
foids
foils
foins
foist
folds
foley
folia
folic
folie
folio
folks
folky
folly
fomes
fonda
fonds
fondu
fones
fonly
fonts
foods
foody
fools
foots
footy
foram
foray
forbs
forby
force
fordo
fords
forel
fores
forex
forge
forgo
forks
forky
forme
forms
forte
forth
forts
forty
forum
forza
forze
fossa
fosse
fouat
fouds
fouer
fouet
foule
fouls
found
fount
fours
fouth
fovea
fowls
fowth
foxed
foxes
foxie
foyer
foyle
foyne
frabs
frack
fract
frags
frail
fraim
frame
franc
frank
frape
fraps
frass
frate
frati
frats
fraud
fraus
frays
freak
freed
freer
frees
freet
freit
fremd
frena
freon
frere
fresh
frets
friar
fribs
fried
frier
fries
frigs
frill
frise
frisk
frist
frith
frits
fritt
fritz
frize
frizz
frock
froes
frogs
frond
frons
front
frore
frorn
frory
frosh
frost
froth
frown
frows
frowy
froze
frugs
fruit
frump
frush
frust
fryer
fubar
fubby
fubsy
fucks
fucus
fuddy
fudge
fudgy
fuels
fuero
fuffs
fuffy
fugal
fuggy
fugie
fugio
fugle
fugly
fugue
fugus
fujis
fulls
fully
fumed
fumer
fumes
fumet
fundi
funds
fundy
fungi
fungo
fungs
funks
funky
funny
fural
furan
furca
furls
furol
furor
furrs
furry
furth
furze
furzy
fused
fusee
fusel
fuses
fusil
fusks
fussy
fusts
fusty
futon
fuzed
fuzee
fuzes
fuzil
fuzzy
fyces
fyked
fykes
fyles
fyrds
fytte
gabba
gabby
gable
gaddi
gades
gadge
gadid
gadis
gadje
gadjo
gadso
gaffe
gaffs
gaged
gager
gages
gaids
gaily
gains
gairs
gaita
gaits
gaitt
gajos
galah
galas
galax
galea
galed
gales
galls
gally
galop
galut
galvo
gamas
gamay
gamba
gambe
gambo
gambs
gamed
gamer
games
gamey
gamic
gamin
gamma
gamme
gammy
gamps
gamut
ganch
gandy
ganef
ganev
gangs
ganja
ganof
gants
gaols
gaped
gaper
gapes
gapos
gappy
garbe
garbo
garbs
garda
gares
garis
garms
garni
garre
garth
garum
gases
gasps
gaspy
gassy
gasts
gatch
gated
gater
gates
gaths
gator
gauch
gaucy
gauds
gaudy
gauge
gauje
gault
gaums
gaumy
gaunt
gaups
gaurs
gauss
gauze
gauzy
gavel
gavot
gawcy
gawds
gawks
gawky
gawps
gawsy
gayal
gayer
gayly
gazal
gazar
gazed
gazer
gazes
gazon
gazoo
geals
geans
geare
gears
geats
gebur
gecko
gecks
geeks
geeky
geeps
geese
geest
geist
geits
gelds
gelee
gelid
gelly
gelts
gemel
gemma
gemmy
gemot
genal
genas
genes
genet
genic
genie
genii
genip
genny
genoa
genom
genre
genro
gents
genty
genua
genus
geode
geoid
gerah
gerbe
geres
gerle
germs
germy
gerne
gesse
gesso
geste
gests
getas
getup
geums
geyan
geyer
ghast
ghats
ghaut
ghazi
ghees
ghest
ghost
ghoul
ghyll
giant
gibed
gibel
giber
gibes
gibli
gibus
giddy
gifts
gigas
gighe
gigot
gigue
gilas
gilds
gilet
gills
gilly
gilpy
gilts
gimel
gimme
gimps
gimpy
ginch
ginge
gings
ginks
ginny
ginzo
gipon
gippo
gippy
gipsy
girds
girls
girly
girns
giron
giros
girrs
girsh
girth
girts
gismo
gisms
gists
gitch
gites
giust
gived
given
giver
gives
gizmo
glace
glade
glads
glady
glaik
glair
glams
gland
glans
glare
glary
glass
glaum
glaur
glaze
glazy
gleam
glean
gleba
glebe
gleby
glede
gleds
gleed
gleek
glees
gleet
gleis
glens
glent
gleys
glial
glias
glibs
glide
gliff
glift
glike
glime
glims
glint
glisk
glits
glitz
gloam
gloat
globe
globi
globs
globy
glode
glogg
gloms
gloom
gloop
glops
glory
gloss
glost
glout
glove
glows
gloze
glued
gluer
glues
gluey
glugs
glume
glums
gluon
glute
gluts
glyph
gnarl
gnarr
gnars
gnash
gnats
gnawn
gnaws
gnome
gnows
goads
goafs
goals
goary
goats
goaty
goban
gobar
gobbi
gobbo
gobby
gobis
gobos
godet
godly
godso
goels
goers
goest
goeth
goety
gofer
goffs
gogga
gogos
goier
going
gojis
golds
goldy
golem
goles
golfs
golly
golpe
golps
gombo
gomer
gompa
gonad
gonch
gonef
goner
gongs
gonia
gonif
gonks
gonna
gonof
gonys
gonzo
gooby
goods
goody
gooey
goofs
goofy
googs
gooks
gooky
goold
gools
gooly
goons
goony
goops
goopy
goors
goory
goose
goosy
gopak
gopik
goral
goras
gored
gores
gorge
goris
gorms
gormy
gorps
gorse
gorsy
gosht
gosse
gotch
goths
gothy
gotta
gouch
gouge
gouks
goura
gourd
gouts
gouty
gowan
gowds
gowfs
gowks
gowls
gowns
goxes
goyim
goyle
graal
grabs
grace
grade
grads
graff
graft
grail
grain
graip
grama
grame
gramp
grams
grana
grand
grans
grant
grape
graph
grapy
grasp
grass
grate
grave
gravs
gravy
grays
graze
great
grebe
grebo
grece
greed
greek
green
grees
greet
grege
grego
grein
grens
grese
greve
grews
greys
grice
gride
grids
grief
griff
grift
grigs
grike
grill
grime
grimy
grind
grins
griot
gripe
grips
gript
gripy
grise
grist
grisy
grith
grits
grize
groan
groat
grody
grogs
groin
groks
groma
grone
groof
groom
grope
gross
grosz
grots
grouf
group
grout
grove
grovy
growl
grown
grows
grrls
grrrl
grubs
grued
gruel
grues
grufe
gruff
grume
grump
grund
grunt
gryce
gryde
gryke
grype
grypt
guaco
guana
guano
guans
guard
guars
guava
gucks
gucky
gudes
guess
guest
guffs
gugas
guide
guids
guild
guile
guilt
guimp
guiro
guise
gulag
gular
gulas
gulch
gules
gulet
gulfs
gulfy
gulls
gully
gulph
gulps
gulpy
gumbo
gumma
gummi
gummy
gumps
gundy
gunge
gungy
gunks
gunky
gunny
guppy
guqin
gurdy
gurge
gurls
gurly
gurns
gurry
gursh
gurus
gushy
gusla
gusle
gusli
gussy
gusto
gusts
gusty
gutsy
gutta
gutty
guyed
guyle
guyot
guyse
gwine
gyals
gyans
gybed
gybes
gyeld
gymps
gynae
gynie
gynny
gynos
gyoza
gypos
gyppo
gyppy
gypsy
gyral
gyred
gyres
gyron
gyros
gyrus
gytes
gyved
gyves
haafs
haars
habit
hable
habus
hacek
hacks
hadal
haded
hades
hadji
hadst
haems
haets
haffs
hafiz
hafts
haggs
hahas
haick
haika
haiks
haiku
hails
haily
hains
haint
hairs
hairy
haith
hajes
hajis
hajji
hakam
hakas
hakea
hakes
hakim
hakus
halal
haled
haler
hales
halfa
halfs
halid
hallo
halls
halma
halms
halon
halos
halse
halts
halva
halve
halwa
hamal
hamba
hamed
hames
hammy
hamza
hanap
hance
hanch
hands
handy
hangi
hangs
hanks
hanky
hansa
hanse
hants
haole
haoma
hapax
haply
happi
happy
hapus
haram
hards
hardy
hared
harem
hares
harim
harks
harls
harms
harns
haros
harps
harpy
harry
harsh
harts
hashy
hasks
hasps
hasta
haste
hasty
hatch
hated
hater
hates
hatha
hauds
haufs
haugh
hauld
haulm
hauls
hault
hauns
haunt
hause
haute
haven
haver
haves
havoc
hawed
hawks
hawms
hawse
hayed
hayer
hayey
hayle
hazan
hazed
hazel
hazer
hazes
heads
heady
heald
heals
heame
heaps
heapy
heard
heare
hears
heart
heast
heath
heats
heave
heavy
heben
hebes
hecht
hecks
heder
hedge
hedgy
heeds
heedy
heels
heeze
hefte
hefts
hefty
heids
heigh
heils
heirs
heist
hejab
hejra
heled
heles
helio
helix
hello
hells
helms
helos
helot
helps
helve
hemal
hemes
hemic
hemin
hemps
hempy
hence
hench
hends
henge
henna
henny
henry
hents
hepar
herbs
herby
herds
heres
herls
herma
herms
herns
heron
heros
herry
herse
hertz
herye
hesps
hests
hetes
heths
heuch
heugh
hevea
hewed
hewer
hewgh
hexad
hexed
hexer
hexes
hexyl
heyed
hiant
hicks
hided
hider
hides
hiems
highs
hight
hijab
hijra
hiked
hiker
hikes
hikoi
hilar
hilch
hillo
hills
hilly
hilts
hilum
hilus
himbo
hinau
hinds
hinge
hings
hinky
hinny
hints
hiois
hiply
hippo
hippy
hired
hiree
hirer
hires
hissy
hists
hitch
hithe
hived
hiver
hives
hizen
hoaed
hoagy
hoard
hoars
hoary
hoast
hobby
hobos
hocks
hocus
hodad
hodja
hoers
hogan
hogen
hoggs
hoghs
hohed
hoick
hoied
hoiks
hoing
hoise
hoist
hokas
hoked
hokes
hokey
hokis
hokku
hokum
holds
holed
holes
holey
holks
holla
hollo
holly
holme
holms
holon
holos
holts
homas
homed
homer
homes
homey
homie
homme
homos
honan
honda
honds
honed
honer
hones
honey
hongi
hongs
honks
honky
honor
hooch
hoods
hoody
hooey
hoofs
hooka
hooks
hooky
hooly
hoons
hoops
hoord
hoors
hoosh
hoots
hooty
hoove
hopak
hoped
hoper
hopes
hoppy
horah
horal
horas
horde
horis
horks
horme
horns
horny
horse
horst
horsy
hosed
hosel
hosen
hoser
hoses
hosey
hosta
hosts
hotch
hotel
hoten
hotly
hotty
houff
houfs
hough
hound
houri
hours
house
houts
hovea
hoved
hovel
hoven
hover
hoves
howbe
howdy
howes
howff
howfs
howks
howls
howre
howso
hoxed
hoxes
hoyas
hoyed
hoyle
hubby
hucks
hudna
hudud
huers
huffs
huffy
huger
huggy
huhus
huias
hulas
hules
hulks
hulky
hullo
hulls
hully
human
humas
humfs
humic
humid
humor
humph
humps
humpy
humus
hunch
hunks
hunky
hunts
hurds
hurls
hurly
hurra
hurry
hurst
hurts
hushy
husks
husky
husos
hussy
hutch
hutia
huzza
huzzy
hwyls
hydra
hydro
hyena
hyens
hygge
hying
hykes
hylas
hyleg
hyles
hylic
hymen
hymns
hynde
hyoid
hyped
hyper
hypes
hypha
hyphy
hypos
hyrax
hyson
hythe
iambi
iambs
ibrik
icers
iched
iches
ichor
icier
icily
icing
icker
ickle
icons
ictal
ictic
ictus
idant
ideal
ideas
idees
ident
idiom
idiot
idled
idler
idles
idola
idols
idyll
idyls
iftar
igapo
igged
igloo
iglus
ihram
ikans
ikats
ikons
ileac
ileal
ileum
ileus
iliac
iliad
ilial
ilium
iller
illth
image
imago
imams
imari
imaum
imbar
imbed
imbue
imide
imido
imids
imine
imino
immew
immit
immix
imped
impel
impis
imply
impot
impro
imshi
imshy
inane
inapt
inarm
inbox
inbye
incel
incle
incog
incur
incus
incut
indew
index
india
indie
indol
indow
indri
indue
inept
inerm
inert
infer
infix
infos
infra
ingan
ingle
ingot
inion
inked
inker
inkle
inlay
inlet
inned
inner
innit
inorb
input
inrun
inset
inspo
intel
inter
intil
intis
intra
intro
inula
inure
inurn
inust
invar
inwit
iodic
iodid
iodin
ionic
iotas
ippon
irade
irate
irids
iring
irked
iroko
irone
irons
irony
isbas
ishes
isled
isles
islet
isnae
issei
issue
istle
itchy
items
ither
ivied
ivies
ivory
ixias
ixnay
ixora
ixtle
izard
izars
izzat
jaaps
jabot
jacal
jacks
jacky
jaded
jades
jafas
jaffa
jagas
jager
jaggs
jaggy
jagir
jagra
jails
jaker
jakes
jakey
jalap
jalop
jambe
jambo
jambs
jambu
james
jammy
jamon
janes
janns
janny
janty
japan
japed
japer
japes
jarks
jarls
jarps
jarta
jarul
jasey
jaspe
jasps
jatos
jauks
jaunt
jaups
javas
javel
jawan
jawed
jaxie
jazzy
jeans
jeats
jebel
jedis
jeels
jeely
jeeps
jeers
jeeze
jefes
jeffs
jehad
jehus
jelab
jello
jells
jelly
jembe
jemmy
jenny
jeons
jerid
jerks
jerky
jerry
jesse
jests
jesus
jetes
jeton
jetty
jeune
jewed
jewel
jewie
jhala
jiaos
jibba
jibbs
jibed
jiber
jibes
jiffs
jiffy
jiggy
jigot
jihad
jills
jilts
jimmy
jimpy
jingo
jinks
jinne
jinni
jinns
jirds
jirga
jirre
jisms
jived
jiver
jives
jivey
jnana
jobed
jobes
jocko
jocks
jocky
jocos
jodel
joeys
johns
joins
joint
joist
joked
joker
jokes
jokey
jokol
joled
joles
jolls
jolly
jolts
jolty
jomon
jomos
jones
jongs
jonty
jooks
joram
jorum
jotas
jotty
jotun
joual
jougs
jouks
joule
jours
joust
jowar
jowed
jowls
jowly
joyed
jubas
jubes
jucos
judas
judge
judgy
judos
jugal
jugum
juice
juicy
jujus
juked
jukes
jukus
julep
jumar
jumbo
jumby
jumps
jumpy
junco
junks
junky
junta
junto
jupes
jupon
jural
jurat
jurel
jures
juror
justs
jutes
jutty
juves
juvie
kaama
kabab
kabar
kabob
kacha
kacks
kadai
kades
kadis
kafir
kagos
kagus
kahal
kaiak
kaids
kaies
kaifs
kaika
kaiks
kails
kaims
kaing
kains
kakas
kakis
kalam
kales
kalif
kalis
kalpa
kamas
kames
kamik
kamis
kamme
kanae
kanas
kandy
kaneh
kanes
kanga
kangs
kanji
kants
kanzu
kaons
kapas
kaphs
kapok
kapow
kappa
kapus
kaput
karas
karat
karks
karma
karns
karoo
karos
karri
karst
karsy
karts
karzy
kasha
kasme
katal
katas
katis
katti
kaugh
kauri
kauru
kaury
kaval
kavas
kawas
kawau
kawed
kayak
kayle
kayos
kazis
kazoo
kbars
kebab
kebar
kebob
kecks
kedge
kedgy
keech
keefs
keeks
keels
keema
keeno
keens
keeps
keets
keeve
kefir
kehua
keirs
kelep
kelim
kells
kelly
kelps
kelpy
kelts
kelty
kembo
kembs
kemps
kempt
kempy
kenaf
kench
kendo
kenos
kente
kents
kepis
kerbs
kerel
kerfs
kerky
kerma
kerne
kerns
keros
kerry
kerve
kesar
kests
ketas
ketch
ketes
ketol
kevel
kevil
kexes
keyed
keyer
khadi
khafs
khaki
khans
khaph
khats
khaya
khazi
kheda
kheth
khets
khoja
khors
khoum
khuds
kiaat
kiack
kiang
kibbe
kibbi
kibei
kibes
kibla
kicks
kicky
kiddo
kiddy
kidel
kidge
kiefs
kiers
kieve
kievs
kight
kikes
kikoi
kiley
kilim
kills
kilns
kilos
kilps
kilts
kilty
kimbo
kinas
kinda
kinds
kindy
kines
kings
kinin
kinks
kinky
kinos
kiore
kiosk
kipes
kippa
kipps
kirby
kirks
kirns
kirri
kisan
kissy
kists
kited
kiter
kites
kithe
kiths
kitty
kitul
kivas
kiwis
klang
klaps
klett
klick
klieg
kliks
klong
kloof
kluge
klutz
knack
knags
knaps
knarl
knars
knaur
knave
knawe
knead
kneed
kneel
knees
knell
knelt
knife
knish
knits
knive
knobs
knock
knoll
knops
knosp
knots
knout
knowe
known
knows
knubs
knurl
knurr
knurs
knuts
koala
koans
koaps
koban
kobos
koels
koffs
kofta
kogal
kohas
kohen
kohls
koine
kojis
kokam
kokas
koker
kokra
kokum
kolas
kolos
kombu
konbu
kondo
konks
kooks
kooky
koori
kopek
kophs
kopje
koppa
korai
koran
koras
korat
kores
korma
koros
korun
korus
koses
kotch
kotos
kotow
koura
kraal
krabs
kraft
krais
krait
krang
krans
kranz
kraut
krays
kreep
kreng
krewe
krill
krona
krone
kroon
krubi
krunk
ksars
kubie
kudos
kudus
kudzu
kufis
kugel
kuias
kukri
kukus
kulak
kulan
kulas
kulfi
kumis
kumys
kuris
kurre
kurta
kurus
kusso
kutas
kutch
kutis
kutus
kuzus
kvass
kvell
kwela
kyack
kyaks
kyang
kyars
kyats
kybos
kydst
kyles
kylie
kylin
kylix
kyloe
kynde
kynds
kypes
kyrie
kytes
kythe
laari
labda
label
labia
labis
labor
labra
laced
lacer
laces
lacet
lacey
lacks
laddy
laded
laden
lader
lades
ladle
laers
laevo
lagan
lager
lahal
lahar
laich
laics
laids
laigh
laika
laiks
laird
lairs
lairy
laith
laity
laked
laker
lakes
lakhs
lakin
laksa
laldy
lalls
lamas
lambs
lamby
lamed
lamer
lames
lamia
lammy
lamps
lanai
lanas
lance
lanch
lande
lands
lanes
lanks
lanky
lants
lapel
lapin
lapis
lapje
lapse
larch
lards
lardy
laree
lares
large
largo
laris
larks
larky
larns
larnt
larum
larva
lased
laser
lases
lassi
lasso
lassu
lassy
lasts
latah
latch
lated
laten
later
latex
lathe
lathi
laths
lathy
latke
latte
latus
lauan
lauch
lauds
laufs
laugh
laund
laura
laval
lavas
laved
laver
laves
lavra
lavvy
lawed
lawer
lawin
lawks
lawns
lawny
laxed
laxer
laxes
laxly
layed
layer
layin
layup
lazar
lazed
lazes
lazos
lazzi
lazzo
leach
leads
leady
leafs
leafy
leaks
leaky
leams
leans
leant
leany
leaps
leapt
leare
learn
lears
leary
lease
leash
least
leats
leave
leavy
leaze
leben
leccy
ledes
ledge
ledgy
ledum
leear
leech
leeks
leeps
leers
leery
leese
leets
leeze
lefte
lefts
lefty
legal
leger
leges
legge
leggo
leggy
legit
lehrs
lehua
leirs
leish
leman
lemed
lemel
lemes
lemma
lemme
lemon
lemur
lends
lenes
lengs
lenis
lenos
lense
lenti
lento
leone
leper
lepid
lepra
lepta
lered
leres
lerps
lesbo
leses
lests
letch
lethe
letup
leuch
leuco
leuds
leugh
levas
levee
level
lever
leves
levin
levis
lewis
lexes
lexis
lezes
lezza
lezzy
liana
liane
liang
liard
liars
liart
libel
liber
libra
libri
lichi
licht
licit
licks
lidar
lidos
liefs
liege
liens
liers
lieus
lieve
lifer
lifes
lifts
ligan
liger
ligge
light
ligne
liked
liken
liker
likes
likin
lilac
lills
lilos
lilts
liman
limas
limax
limba
limbi
limbo
limbs
limby
limed
limen
limes
limey
limit
limma
limns
limos
limpa
limps
linac
linch
linds
lindy
lined
linen
liner
lines
liney
linga
lingo
lings
lingy
linin
links
linky
linns
linny
linos
lints
linty
linum
linux
lions
lipas
lipes
lipid
lipin
lipos
lippy
liras
lirks
lirot
lisks
lisle
lisps
lists
litai
litas
lited
liter
lites
lithe
litho
liths
litre
lived
liven
liver
lives
livid
livor
livre
llama
llano
loach
loads
loafs
loams
loamy
loans
loast
loath
loave
lobar
lobby
lobed
lobes
lobos
lobus
local
loche
lochs
locie
locis
locks
locos
locum
locus
loden
lodes
lodge
loess
lofts
lofty
logan
loges
loggy
logia
logic
logie
login
logoi
logon
logos
lohan
loids
loins
loipe
loirs
lokes
lolls
lolly
lolog
lomas
lomed
lomes
loner
longa
longe
longs
looby
looed
looey
loofa
loofs
looie
looks
looky
looms
loons
loony
loops
loopy
loord
loose
loots
loped
loper
lopes
loppy
loral
loran
lords
lordy
lorel
lores
loric
loris
lorry
losed
losel
losen
loser
loses
lossy
lotah
lotas
lotes
lotic
lotos
lotsa
lotta
lotte
lotto
lotus
loued
lough
louie
louis
louma
lound
louns
loupe
loups
loure
lours
loury
louse
lousy
louts
lovat
loved
lover
loves
lovey
lovie
lowan
lowed
lower
lowes
lowly
lownd
lowne
lowns
lowps
lowry
lowse
lowts
loxed
loxes
loyal
lozen
luach
luaus
lubed
lubes
lubra
luces
lucid
lucks
lucky
lucre
ludes
ludic
ludos
luffa
luffs
luged
luger
luges
lulls
lulus
lumas
lumbi
lumen
lumme
lummy
lumps
lumpy
lunar
lunas
lunch
lunes
lunet
lunge
lungi
lungs
lunks
lunts
lupin
lupus
lurch
lured
lurer
lures
lurex
lurgi
lurgy
lurid
lurks
lurry
lurve
luser
lushy
lusks
lusts
lusty
lusus
lutea
luted
luter
lutes
luvvy
luxed
luxer
luxes
lweis
lyams
lyard
lyart
lyase
lycea
lycee
lycra
lying
lymes
lymph
lynch
lynes
lyres
lyric
lysed
lyses
lysin
lysis
lysol
lyssa
lyted
lytes
lythe
lytic
lytta
maaed
maare
maars
mabes
macas
macaw
maced
macer
maces
mache
machi
macho
machs
macks
macle
macon
macro
madam
madge
madid
madly
madre
maerl
mafia
mafic
mages
maggs
magic
magma
magot
magus
mahoe
mahua
mahwa
maids
maiko
maiks
maile
maill
mails
maims
mains
maire
mairs
maise
maist
maize
major
makar
maker
makes
makis
makos
malam
malar
malas
malax
males
malic
malik
malis
malls
malms
malmy
malts
malty
malus
malva
malwa
mamas
mamba
mambo
mamee
mamey
mamie
mamma
mammy
manas
manat
mandi
maneb
maned
maneh
manes
manet
manga
mange
mango
mangs
mangy
mania
manic
manis
manky
manly
manna
manor
manos
manse
manta
manto
manty
manul
manus
mapau
maple
maqui
marae
marah
maras
march
marcs
mardy
mares
marge
margs
maria
marid
marka
marks
marle
marls
marly
marms
maron
maror
marra
marri
marry
marse
marsh
marts
marvy
masas
mased
maser
mases
mashy
masks
mason
massa
masse
massy
masts
masty
masus
matai
match
mated
mater
mates
matey
maths
matin
matlo
matte
matts
matza
matzo
mauby
mauds
mauls
maund
mauri
mausy
mauts
mauve
mauzy
maven
mavie
mavin
mavis
mawed
mawks
mawky
mawns
mawrs
maxed
maxes
maxim
maxis
mayan
mayas
maybe
mayed
mayor
mayos
mayst
mazed
mazer
mazes
mazey
mazut
mbira
meads
meals
mealy
meane
means
meant
meany
meare
mease
meath
meats
meaty
mebos
mecca
mechs
mecks
medal
media
medic
medii
medle
meeds
meers
meets
meffs
meins
meint
meiny
meith
mekka
melas
melba
melds
melee
melic
melik
mells
melon
melts
melty
memes
memos
menad
mends
mened
menes
menge
mengs
mensa
mense
mensh
menta
mento
menus
meous
meows
merch
mercs
mercy
merde
mered
merel
merer
meres
merge
meril
meris
merit
merks
merle
merls
merry
merse
mesal
mesas
mesel
meses
meshy
mesic
mesne
meson
messy
mesto
metal
meted
meter
metes
metho
meths
metic
metif
metis
metol
metre
metro
meuse
meved
meves
mewed
mewls
meynt
mezes
mezze
mezzo
mhorr
miaou
miaow
miasm
miaul
micas
miche
micht
micks
micky
micos
micra
micro
middy
midge
midgy
midis
midst
miens
mieve
miffs
miffy
mifty
miggs
might
mihas
mihis
miked
mikes
mikra
mikva
milch
milds
miler
miles
milfs
milia
milko
milks
milky
mille
mills
milor
milos
milpa
milts
milty
miltz
mimed
mimeo
mimer
mimes
mimic
mimsy
minae
minar
minas
mince
mincy
minds
mined
miner
mines
minge
mings
mingy
minim
minis
minke
minks
minny
minor
minos
mints
minty
minus
mired
mires
mirex
mirid
mirin
mirks
mirky
mirly
miros
mirth
mirvs
mirza
misch
misdo
miser
mises
misgo
misos
missa
missy
mists
misty
mitch
miter
mites
mitis
mitre
mitts
mixed
mixen
mixer
mixes
mixte
mixup
mizen
mizzy
mneme
moans
moats
mobby
mobes
mobey
mobie
moble
mocha
mochi
mochs
mochy
mocks
modal
model
modem
moder
modes
modge
modii
modus
moers
mofos
moggy
mogul
mohel
mohos
mohrs
mohua
mohur
moile
moils
moira
moire
moist
moits
mojos
mokes
mokis
mokos
molal
molar
molas
molds
moldy
moled
moles
molla
molls
molly
molto
molts
molys
momes
momma
mommy
momus
monad
monal
monas
monde
mondo
moner
money
mongo
mongs
monic
monie
monks
monos
monte
month
monty
moobs
mooch
moods
moody
mooed
mooks
moola
mooli
mools
mooly
moong
moons
moony
moops
moors
moory
moose
moots
moove
moped
moper
mopes
mopey
moppy
mopsy
mopus
morae
moral
moras
morat
moray
morel
mores
moria
morne
morns
moron
morph
morra
morro
morse
morts
mosed
moses
mosey
mosks
mosso
mossy
moste
mosts
moted
motel
moten
motes
motet
motey
moths
mothy
motif
motis
motor
motte
motto
motts
motty
motus
motza
mouch
moues
mould
mouls
moult
mound
mount
moups
mourn
mouse
moust
mousy
mouth
moved
mover
moves
movie
mowas
mowed
mower
mowra
moxas
moxie
moyas
moyle
moyls
mozed
mozes
mozos
mpret
mucho
mucic
mucid
mucin
mucks
mucky
mucor
mucro
mucus
muddy
mudge
mudir
mudra
muffs
mufti
mugga
muggs
muggy
muhly
muids
muils
muirs
muist
mujik
mulch
mulct
muled
mules
muley
mulga
mulie
mulla
mulls
mulse
mulsh
mumms
mummy
mumps
mumsy
mumus
munch
munga
munge
mungo
mungs
munis
munts
muntu
muons
mural
muras
mured
mures
murex
murid
murks
murky
murls
murly
murra
murre
murri
murrs
murry
murti
murva
musar
musca
mused
muser
muses
muset
musha
mushy
music
musit
musks
musky
musos
musse
mussy
musth
musts
musty
mutch
muted
muter
mutes
mutha
mutis
muton
mutts
muxed
muxes
muzak
muzzy
mvule
myall
mylar
mynah
mynas
myoid
myoma
myope
myops
myopy
myrrh
mysid
mythi
myths
mythy
myxos
mzees
naams
naans
nabes
nabis
nabks
nabla
nabob
nache
nacho
nacre
nadas
nadir
naeve
naevi
naffs
nagas
naggy
nagor
nahal
naiad
naifs
naiks
nails
naira
nairu
naive
naked
naker
nakfa
nalas
naled
nalla
named
namer
names
namma
namus
nanas
nance
nancy
nandu
nanna
nanny
nanos
nanua
napas
naped
napes
napoo
nappa
nappe
nappy
naras
narco
narcs
nards
nares
naric
naris
narks
narky
narre
nasal
nashi
nasty
natal
natch
nates
natis
natty
nauch
naunt
naval
navar
navel
naves
navew
navvy
nawab
nazes
nazir
nazis
nduja
neafe
neals
neaps
nears
neath
neats
nebek
nebel
necks
neddy
needs
needy
neeld
neele
neemb
neems
neeps
neese
neeze
negro
negus
neifs
neigh
neist
neive
nelis
nelly
nemas
nemns
nempt
nenes
neons
neper
nepit
neral
nerds
nerdy
nerka
nerks
nerol
nerts
nertz
nerve
nervy
nests
netes
netop
netts
netty
neuks
neume
neums
nevel
never
neves
nevus
newbs
newed
newel
newer
newie
newly
newsy
newts
nexts
nexus
ngaio
ngana
ngati
ngoma
ngwee
nicad
nicer
niche
nicht
nicks
nicol
nidal
nided
nides
nidor
nidus
niece
niefs
nieve
nifes
niffs
niffy
nifty
niger
nighs
night
nihil
nikab
nikah
nikau
nills
nimbi
nimbs
nimps
niner
nines
ninja
ninny
ninon
ninth
nipas
nippy
niqab
nirls
nirly
nisei
nisse
nisus
niter
nites
nitid
niton
nitre
nitro
nitry
nitty
nival
nixed
nixer
nixes
nixie
nizam
nkosi
noahs
nobby
noble
nobly
nocks
nodal
noddy
nodes
nodus
noels
noggs
nohow
noils
noily
noint
noirs
noise
noisy
noles
nolls
nolos
nomad
nomas
nomen
nomes
nomic
nomoi
nomos
nonas
nonce
nones
nonet
nongs
nonis
nonny
nonyl
noobs
nooit
nooks
nooky
noons
noops
noose
nopal
noria
noris
norks
norma
norms
north
nosed
noser
noses
nosey
notal
notch
noted
noter
notes
notum
nould
noule
nouls
nouns
nouny
noups
novae
novas
novel
novum
noway
nowed
nowls
nowts
nowty
noxal
noxes
noyau
noyed
noyes
nubby
nubia
nucha
nuddy
nuder
nudes
nudge
nudie
nudzh
nuffs
nugae
nuked
nukes
nulla
nulls
numbs
numen
nummy
nunny
nurds
nurdy
nurls
nurrs
nurse
nutso
nutsy
nutty
nyaff
nyala
nying
nylon
nymph
nyssa
oaked
oaken
oaker
oakum
oared
oases
oasis
oasts
oaten
oater
oaths
oaves
obang
obeah
obeli
obese
obeys
obias
obied
obiit
obits
objet
oboes
obole
oboli
obols
occam
occur
ocean
ocher
oches
ochre
ochry
ocker
ocrea
octad
octal
octan
octas
octet
octyl
oculi
odahs
odals
odder
oddly
odeon
odeum
odism
odist
odium
odors
odour
odyle
odyls
ofays
offal
offed
offer
offie
oflag
often
ofter
ogams
ogeed
ogees
oggin
ogham
ogive
ogled
ogler
ogles
ogmic
ogres
ohias
ohing
ohmic
ohone
oidia
oiled
oiler
oinks
oints
ojime
okapi
okays
okehs
okras
oktas
olden
older
oldie
oleic
olein
olent
oleos
oleum
olios
olive
ollas
ollav
oller
ollie
ology
olpae
olpes
omasa
omber
ombre
ombus
omega
omens
omers
omits
omlah
omovs
omrah
oncer
onces
oncet
oncus
onely
oners
onery
onion
onium
onkus
onlay
onned
onset
ontic
oobit
oohed
oomph
oonts
ooped
oorie
ooses
ootid
oozed
oozes
opahs
opals
opens
opepe
opera
opine
oping
opium
oppos
opsin
opted
opter
optic
orach
oracy
orals
orang
orant
orate
orbed
orbit
orcas
orcin
order
ordos
oread
orfes
organ
orgia
orgic
orgue
oribi
oriel
orixa
orles
orlon
orlop
ormer
ornis
orpin
orris
ortho
orval
orzos
oscar
oshac
osier
osmic
osmol
ossia
ostia
otaku
otary
other
ottar
otter
ottos
oubit
oucht
ouens
ought
ouija
oulks
oumas
ounce
oundy
oupas
ouped
ouphe
ouphs
ourie
ousel
ousts
outby
outdo
outed
outer
outgo
outre
outro
outta
ouzel
ouzos
ovals
ovary
ovate
ovels
ovens
overs
overt
ovine
ovist
ovoid
ovoli
ovolo
ovule
owche
owies
owing
owled
owler
owlet
owned
owner
owres
owrie
owsen
oxbow
oxers
oxeye
oxide
oxids
oxies
oxime
oxims
oxlip
oxter
oyers
ozeki
ozone
ozzie
paals
paans
pacas
paced
pacer
paces
pacey
pacha
packs
pacos
pacta
pacts
paddy
padis
padle
padma
padre
padri
paean
paedo
paeon
pagan
paged
pager
pages
pagle
pagod
pagri
paiks
pails
pains
paint
paire
pairs
paisa
paise
pakka
palas
palay
palea
paled
paler
pales
palet
palis
palki
palla
palls
pally
palms
palmy
palpi
palps
palsa
palsy
pampa
panax
pance
panda
pands
pandy
paned
panel
panes
panga
pangs
panic
panim
panko
panne
panni
pansy
panto
pants
panty
paoli
paolo
papal
papas
papaw
paper
papes
pappi
pappy
parae
paras
parch
pardi
pards
pardy
pared
paren
pareo
parer
pares
pareu
parev
parge
pargo
paris
parka
parki
parks
parky
parle
parly
parma
parol
parps
parra
parrs
parry
parse
parti
parts
party
parve
parvo
paseo
pases
pasha
pashm
paska
paspy
passe
pasta
paste
pasts
pasty
patch
pated
paten
pater
pates
paths
patin
patio
patka
patly
patsy
patte
patty
patus
pauas
pauls
pause
pavan
paved
paven
paver
paves
pavid
pavin
pavis
pawas
pawaw
pawed
pawer
pawks
pawky
pawls
pawns
paxes
payed
payee
payer
payor
paysd
peace
peach
peage
peags
peaks
peaky
peals
peans
peare
pearl
pears
peart
pease
peats
peaty
peavy
peaze
pebas
pecan
pechs
pecke
pecks
pecky
pedal
pedes
pedis
pedro
peece
peeks
peels
peens
peeoy
peepe
peeps
peers
peery
peeve
peggy
peghs
peins
peise
peize
pekan
pekes
pekin
pekoe
pelas
pelau
peles
pelfs
pells
pelma
pelon
pelta
pelts
penal
pence
pends
pendu
pened
penes
pengo
penie
penis
penks
penna
penne
penni
penny
pents
peons
peony
pepla
pepos
peppy
pepsi
perai
perce
perch
percs
perdu
perdy
perea
peres
peril
peris
perks
perky
perms
perns
perog
perps
perry
perse
perst
perts
perve
pervo
pervs
pervy
pesky
pesos
pesto
pests
pesty
petal
petar
peter
petit
petre
petri
petti
petto
petty
pewee
pewit
peyse
phage
phang
phare
pharm
phase
pheer
phene
pheon
phese
phial
phish
phizz
phlox
phoca
phone
phono
phons
phony
photo
phots
phpht
phuts
phyla
phyle
piani
piano
pians
pibal
pical
picas
piccy
picks
picky
picot
picra
picul
piece
piend
piers
piert
pieta
piets
piety
piezo
piggy
pight
pigmy
piing
pikas
pikau
piked
piker
pikes
pikey
pikis
pikul
pilae
pilaf
pilao
pilar
pilau
pilaw
pilch
pilea
piled
pilei
piler
piles
pilis
pills
pilot
pilow
pilum
pilus
pimas
pimps
pinas
pinch
pined
pines
piney
pingo
pings
pinko
pinks
pinky
pinna
pinny
pinon
pinot
pinta
pinto
pints
pinup
pions
piony
pious
pioye
pioys
pipal
pipas
piped
piper
pipes
pipet
pipis
pipit
pippy
pipul
pique
pirai
pirls
pirns
pirog
pisco
pises
pisky
pisos
pissy
piste
pitas
pitch
piths
pithy
piton
pitot
pitta
piums
pivot
pixel
pixes
pixie
pized
pizes
pizza
plaas
place
plack
plage
plaid
plain
plait
plane
plank
plans
plant
plaps
plash
plasm
plast
plate
plats
platt
platy
playa
plays
plaza
plead
pleas
pleat
plebe
plebs
plena
pleon
plesh
plews
plica
plied
plier
plies
plims
pling
plink
ploat
plods
plong
plonk
plook
plops
plots
plotz
plouk
plows
ploye
ploys
pluck
plues
pluff
plugs
plumb
plume
plump
plums
plumy
plunk
pluot
plush
pluto
plyer
poach
poaka
poake
poboy
pocks
pocky
podal
poddy
podex
podge
podgy
podia
poems
poeps
poesy
poets
pogey
pogge
pogos
pohed
poilu
poind
point
poise
pokal
poked
poker
pokes
pokey
pokie
polar
poled
poler
poles
poley
polio
polis
polje
polka
polks
polls
polly
polos
polts
polyp
polys
pombe
pomes
pommy
pomos
pomps
ponce
poncy
ponds
pones
poney
ponga
pongo
pongs
pongy
ponks
ponts
ponty
ponzu
pooch
poods
pooed
poofs
poofy
poohs
pooja
pooka
pooks
pools
poons
poops
poopy
poori
poort
poots
poove
poovy
popes
poppa
poppy
popsy
porae
poral
porch
pored
porer
pores
porge
porgy
porin
porks
porky
porno
porns
porny
porta
ports
porty
posed
poser
poses
posey
posho
posit
posse
posts
potae
potch
poted
potes
potin
potoo
potsy
potto
potts
potty
pouch
pouff
poufs
pouke
pouks
poule
poulp
poult
pound
poupe
poupt
pours
pouts
pouty
powan
power
powin
pownd
powns
powny
powre
poxed
poxes
poynt
poyou
poyse
pozzy
praam
prads
prahu
prams
prana
prang
prank
praos
prase
prate
prats
pratt
praty
praus
prawn
prays
predy
preed
preen
prees
preif
prems
premy
prent
preon
preop
preps
presa
prese
press
prest
preve
prexy
preys
prial
price
prick
pricy
pride
pried
prief
prier
pries
prigs
prill
prima
prime
primi
primo
primp
prims
primy
prink
print
prion
prior
prise
prism
priss
privy
prize
proas
probe
probs
prods
proem
profs
progs
proin
proke
prole
proll
promo
proms
prone
prong
pronk
proof
props
prore
prose
proso
pross
prost
prosy
proto
proud
proul
prove
prowl
prows
proxy
proyn
prude
prune
prunt
pruta
pryer
pryse
psalm
pseud
pshaw
psion
psoae
psoai
psoas
psora
psych
psyop
pubco
pubes
pubic
pubis
pucan
pucer
puces
pucka
pucks
puddy
pudge
pudgy
pudic
pudor
pudsy
pudus
puers
puffa
puffs
puffy
puggy
pugil
puhas
pujah
pujas
pukas
puked
puker
pukes
pukey
pukka
pukus
pulao
pulas
puled
puler
pules
pulik
pulis
pulka
pulks
pulli
pulls
pully
pulmo
pulps
pulpy
pulse
pulus
pumas
pumie
pumps
punas
punce
punch
punga
pungs
punji
punka
punks
punky
punny
punto
punts
punty
pupae
pupal
pupas
pupil
puppy
pupus
purda
pured
puree
purer
pures
purge
purin
puris
purls
purpy
purrs
purse
pursy
purty
puses
pushy
pusle
pussy
putid
puton
putti
putto
putts
putty
puzel
pwned
pyats
pyets
pygal
pygmy
pyins
pylon
pyned
pynes
pyoid
pyots
pyral
pyran
pyres
pyrex
pyric
pyros
pyxed
pyxes
pyxie
pyxis
pzazz
qadis
qaids
qajaq
qanat
qapik
qibla
qophs
qorma
quack
quads
quaff
quags
quail
quair
quais
quake
quaky
quale
qualm
quant
quare
quark
quart
quash
quasi
quass
quate
quats
quayd
quays
qubit
quean
queen
queer
quell
queme
quena
quern
query
quest
queue
queyn
queys
quich
quick
quids
quiet
quiff
quill
quilt
quims
quina
quine
quino
quins
quint
quipo
quips
quipu
quire
quirk
quirt
quist
quite
quits
quoad
quods
quoif
quoin
quoit
quoll
quonk
quops
quota
quote
quoth
quran
qursh
quyte
rabat
rabbi
rabic
rabid
rabis
raced
racer
races
rache
racks
racon
radar
radge
radii
radio
radix
radon
raffs
rafts
ragas
ragde
raged
ragee
rager
rages
ragga
raggs
raggy
ragis
ragus
rahed
rahui
raias
raids
raiks
raile
rails
raine
rains
rainy
raird
raise
raita
raits
rajah
rajas
rajes
raked
rakee
raker
rakes
rakia
rakis
rakus
rales
rally
ralph
ramal
ramee
ramen
ramet
ramie
ramin
ramis
rammy
ramps
ramus
ranas
rance
ranch
rands
randy
ranee
ranga
range
rangi
rangs
rangy
ranid
ranis
ranke
ranks
rants
raped
raper
rapes
raphe
rapid
rappe
rared
raree
rarer
rares
rarks
rased
raser
rases
rasps
raspy
rasse
rasta
ratal
ratan
ratas
ratch
rated
ratel
rater
rates
ratha
rathe
raths
ratio
ratoo
ratos
ratty
ratus
rauns
raupo
raved
ravel
raven
raver
raves
ravey
ravin
rawer
rawin
rawly
rawns
raxed
raxes
rayah
rayas
rayed
rayle
rayne
rayon
razed
razee
razer
razes
razoo
razor
reach
react
readd
reads
ready
reais
reaks
realm
realo
reals
reame
reams
reamy
reans
reaps
rearm
rears
reast
reata
reate
reave
rebar
rebbe
rebec
rebel
rebid
rebit
rebop
rebus
rebut
rebuy
recal
recap
recce
recco
reccy
recit
recks
recon
recta
recti
recto
recur
recut
redan
redds
reddy
reded
redes
redia
redid
redip
redly
redon
redos
redox
redry
redub
redux
redye
reech
reede
reeds
reedy
reefs
reefy
reeks
reeky
reels
reens
reest
reeve
refed
refel
refer
reffo
refis
refit
refix
refly
refry
regal
regar
reges
reggo
regie
regma
regna
regos
regur
rehab
rehem
reifs
reify
reign
reiki
reiks
reink
reins
reird
reist
reive
rejig
rejon
reked
rekes
rekey
relax
relay
relet
relic
relie
relit
rello
reman
remap
remen
remet
remex
remit
remix
renal
renay
rends
renew
reney
renga
renig
renin
renne
renos
rente
rents
reoil
reorg
repay
repeg
repel
repin
repla
reply
repos
repot
repps
repro
reran
rerig
rerun
resat
resaw
resay
resee
reses
reset
resew
resid
resin
resit
resod
resow
resto
rests
resty
resus
retag
retax
retch
retem
retia
retie
retox
retro
retry
reuse
revel
revet
revie
revue
rewan
rewax
rewed
rewet
rewin
rewon
rewth
rexes
rezes
rheas
rheme
rheum
rhies
rhime
rhine
rhino
rhody
rhomb
rhone
rhumb
rhyme
rhyne
rhyta
riads
rials
riant
riata
ribas
ribby
ribes
riced
ricer
rices
ricey
richt
ricin
ricks
rider
rides
ridge
ridgy
ridic
riels
riems
rieve
rifer
riffs
rifle
rifte
rifts
rifty
riggs
right
rigid
rigol
rigor
riled
riles
riley
rille
rills
rimae
rimed
rimer
rimes
rimus
rinds
rindy
rines
rings
rinks
rinse
rioja
riots
riped
ripen
riper
ripes
ripps
risen
riser
rises
rishi
risks
risky
risps
risus
rites
ritts
ritzy
rival
rivas
rived
rivel
riven
river
rives
rivet
riyal
rizas
roach
roads
roams
roans
roars
roary
roast
roate
robed
robes
robin
roble
robot
rocks
rocky
roded
rodeo
rodes
roger
rogue
roguy
rohes
roids
roils
roily
roins
roist
rojak
rojis
roked
roker
rokes
rolag
roles
rolfs
rolls
romal
roman
romeo
romps
ronde
rondo
roneo
rones
ronin
ronne
ronte
ronts
roods
roofs
roofy
rooks
rooky
rooms
roomy
roons
roops
roopy
roosa
roose
roost
roots
rooty
roped
roper
ropes
ropey
roque
roral
rores
roric
rorid
rorie
rorts
rorty
rosed
roses
roset
roshi
rosin
rosit
rosti
rosts
rotal
rotan
rotas
rotch
roted
rotes
rotis
rotls
roton
rotor
rotos
rotte
rouen
roues
rouge
rough
roule
rouls
roums
round
roups
roupy
rouse
roust
route
routh
routs
roved
roven
rover
roves
rowan
rowdy
rowed
rowel
rowen
rower
rowie
rowme
rownd
rowth
rowts
royal
royne
royst
rozet
rozit
ruana
rubai
rubby
rubel
rubes
rubin
ruble
rubli
rubus
ruche
rucks
rudas
rudds
ruddy
ruder
rudes
rudie
rudis
rueda
ruers
ruffe
ruffs
rugae
rugal
rugby
ruggy
ruing
ruins
rukhs
ruled
ruler
rules
rumal
rumba
rumbo
rumen
rumes
rumly
rummy
rumor
rumpo
rumps
rumpy
runch
runds
runed
runes
rungs
runic
runny
runts
runty
rupee
rupia
rural
rurps
rurus
rusas
ruses
rushy
rusks
rusma
russe
rusts
rusty
ruths
rutin
rutty
ryals
rybat
ryked
rykes
rymme
rynds
ryots
ryper
saags
sabal
sabed
saber
sabes
sabha
sabin
sabir
sable
sabot
sabra
sabre
sacks
sacra
saddo
sades
sadhe
sadhu
sadis
sadly
sados
sadza
safed
safer
safes
sagas
sager
sages
saggy
sagos
sagum
saheb
sahib
saice
saick
saics
saids
saiga
sails
saims
saine
sains
saint
sairs
saist
saith
sajou
sakai
saker
sakes
sakia
sakis
sakti
salad
salal
salat
salep
sales
salet
salic
salix
salle
sally
salmi
salol
salon
salop
salpa
salps
salsa
salse
salto
salts
salty
salue
salut
salve
salvo
saman
samas
samba
sambo
samek
samel
samen
sames
samey
samfu
sammy
sampi
samps
sands
sandy
saned
saner
sanes
sanga
sangh
sango
sangs
sanko
sansa
santo
sants
saola
sapan
sapid
sapor
sappy
saran
sards
sared
saree
sarge
sargo
sarin
saris
sarks
sarky
sarod
saros
sarus
saser
sasin
sasse
sassy
satai
satay
sated
satem
sates
satin
satis
satyr
sauba
sauce
sauch
saucy
saugh
sauls
sault
sauna
saunt
saury
saute
sauts
saved
saver
saves
savey
savin
savor
savoy
savvy
sawah
sawed
sawer
saxes
sayed
sayer
sayid
sayne
sayon
sayst
sazes
scabs
scads
scaff
scags
scail
scala
scald
scale
scall
scalp
scaly
scamp
scams
scand
scans
scant
scapa
scape
scapi
scare
scarf
scarp
scars
scart
scary
scath
scats
scatt
scaud
scaup
scaur
scaws
sceat
scena
scend
scene
scent
schav
schmo
schul
schwa
scion
sclim
scody
scoff
scogs
scold
scone
scoog
scoop
scoot
scopa
scope
scops
score
scorn
scots
scoug
scoup
scour
scout
scowl
scowp
scows
scrab
scrae
scrag
scram
scran
scrap
scrat
scraw
scray
scree
screw
scrim
scrip
scrob
scrod
scrog
scrow
scrub
scrum
scuba
scudi
scudo
scuds
scuff
scuft
scugs
sculk
scull
sculp
sculs
scums
scups
scurf
scurs
scuse
scuta
scute
scuts
scuzz
scyes
sdayn
sdein
seals
seame
seams
seamy
seans
seare
sears
sease
seats
seaze
sebum
secco
sechs
sects
sedan
seder
sedes
sedge
sedgy
sedum
seeds
seedy
seeks
seeld
seels
seely
seems
seeps
seepy
seers
sefer
segar
segni
segno
segol
segos
segue
sehri
seifs
seils
seine
seirs
seise
seism
seity
seiza
seize
sekos
sekts
selah
seles
selfs
sella
selle
sells
selva
semee
semen
semes
semie
semis
senas
sends
senes
sengi
senna
senor
sensa
sense
sensi
sente
senti
sents
senvy
senza
sepad
sepal
sepia
sepic
sepoy
septa
septs
serac
serai
seral
sered
serer
seres
serfs
serge
seric
serif
serin
serks
seron
serow
serra
serre
serrs
serry
serum
serve
servo
sesey
sessa
setae
setal
seton
setts
setup
seven
sever
sewan
sewar
sewed
sewel
sewen
sewer
sewin
sexed
sexer
sexes
sexto
sexts
seyen
shack
shade
shads
shady
shaft
shags
shahs
shake
shako
shakt
shaky
shale
shall
shalm
shalt
shaly
shama
shame
shams
shand
shank
shans
shape
shaps
shard
share
shark
sharn
sharp
shash
shaul
shave
shawl
shawm
shawn
shaws
shaya
shays
shchi
sheaf
sheal
shear
sheas
sheds
sheel
sheen
sheep
sheer
sheet
sheik
shelf
shell
shend
shent
sheol
sherd
shere
shero
shets
sheva
shewn
shews
shiai
shied
shiel
shier
shies
shift
shill
shily
shims
shine
shins
shiny
ships
shire
shirk
shirr
shirs
shirt
shish
shiso
shist
shite
shits
shiur
shiva
shive
shivs
shlep
shlub
shmek
shmoe
shoal
shoat
shock
shoed
shoer
shoes
shogi
shogs
shoji
shojo
shola
shone
shook
shool
shoon
shoos
shoot
shope
shops
shore
shorl
shorn
short
shote
shots
shott
shout
shove
showd
shown
shows
showy
shoyu
shred
shrew
shris
shrow
shrub
shrug
shtik
shtum
shtup
shuck
shule
shuln
shuls
shuns
shunt
shura
shush
shute
shuts
shwas
shyer
shyly
sials
sibbs
sibyl
sices
sicht
sicko
sicks
sicky
sidas
sided
sider
sides
sidha
sidhe
sidle
siege
sield
siens
sient
sieth
sieur
sieve
sifts
sighs
sight
sigil
sigla
sigma
signa
signs
sijos
sikas
siker
sikes
silds
siled
silen
siler
siles
silex
silks
silky
sills
silly
silos
silts
silty
silva
simar
simas
simba
simis
simps
simul
since
sinds
sined
sines
sinew
singe
sings
sinhs
sinks
sinky
sinus
siped
sipes
sippy
sired
siree
siren
sires
sirih
siris
siroc
sirra
sirup
sisal
sises
sissy
sista
sists
sitar
sited
sites
sithe
sitka
situp
situs
siver
sixer
sixes
sixmo
sixte
sixth
sixty
sizar
sized
sizel
sizer
sizes
skags
skail
skald
skank
skart
skate
skats
skatt
skaws
skean
skear
skeds
skeed
skeef
skeen
skeer
skees
skeet
skegg
skegs
skein
skelf
skell
skelm
skelp
skene
skens
skeos
skeps
skers
skets
skews
skids
skied
skier
skies
skiey
skiff
skill
skimo
skimp
skims
skink
skins
skint
skios
skips
skirl
skirr
skirt
skite
skits
skive
skivy
sklim
skoal
skody
skoff
skogs
skols
skool
skort
skosh
skran
skrik
skuas
skugs
skulk
skull
skunk
skyed
skyer
skyey
skyfs
skyre
skyrs
skyte
slabs
slack
slade
slaes
slags
slaid
slain
slake
slams
slane
slang
slank
slant
slaps
slart
slash
slate
slats
slaty
slave
slaws
slays
slebs
sleds
sleek
sleep
sleer
sleet
slept
slews
sleys
slice
slick
slide
slier
slily
slime
slims
slimy
sling
slink
slipe
slips
slipt
slish
slits
slive
sloan
slobs
sloes
slogs
sloid
slojd
slomo
sloom
sloop
sloot
slope
slops
slopy
slorm
slosh
sloth
slots
slove
slows
sloyd
slubb
slubs
slued
slues
sluff
slugs
sluit
slump
slums
slung
slunk
slurb
slurp
slurs
sluse
slush
sluts
slyer
slyly
slype
smaak
smack
smaik
small
smalm
smalt
smarm
smart
smash
smaze
smear
smeek
smees
smeik
smeke
smell
smelt
smerk
smews
smile
smirk
smirr
smirs
smite
smith
smits
smock
smogs
smoke
smoko
smoky
smolt
smoor
smoot
smore
smorg
smote
smout
smowt
smugs
smurs
smush
smuts
snabs
snack
snafu
snags
snail
snake
snaky
snaps
snare
snarf
snark
snarl
snars
snary
snash
snath
snaws
snead
sneak
sneap
snebs
sneck
sneds
sneed
sneer
snees
snell
snibs
snick
snide
snies
sniff
snift
snigs
snipe
snips
snipy
snirt
snits
snobs
snods
snoek
snoep
snogs
snoke
snood
snook
snool
snoop
snoot
snore
snort
snots
snout
snowk
snows
snowy
snubs
snuck
snuff
snugs
snush
snyes
soaks
soaps
soapy
soare
soars
soave
sobas
sober
socas
soces
socko
socks
socle
sodas
soddy
sodic
sodom
sofar
sofas
softa
softs
softy
soger
soggy
sohur
soils
soily
sojas
sojus
sokah
soken
sokes
sokol
solah
solan
solar
solas
solde
soldi
soldo
solds
soled
solei
soler
soles
solid
solon
solos
solum
solus
solve
soman
somas
sonar
sonce
sonde
sones
songs
sonic
sonly
sonne
sonny
sonse
sonsy
sooey
sooks
sooky
soole
sools
sooms
soops
soote
sooth
soots
sooty
sophs
sophy
sopor
soppy
sopra
soral
soras
sorbo
sorbs
sorda
sordo
sords
sored
soree
sorel
sorer
sores
sorex
sorgo
sorns
sorra
sorry
sorta
sorts
sorus
soths
sotol
souce
souct
sough
souks
souls
soums
sound
soups
soupy
sours
souse
south
souts
sowar
sowce
sowed
sower
sowff
sowfs
sowle
sowls
sowms
sownd
sowne
sowps
sowse
sowth
soyas
soyle
soyuz
sozin
space
spacy
spade
spado
spaed
spaer
spaes
spags
spahi
spail
spain
spait
spake
spald
spale
spall
spalt
spams
spane
spang
spank
spans
spard
spare
spark
spars
spart
spasm
spate
spats
spaul
spawl
spawn
spaws
spayd
spays
spaza
spazz
speak
speal
spean
spear
speat
speck
specs
spect
speed
speel
speer
speil
speir
speks
speld
spelk
spell
spelt
spend
spent
speos
sperm
spets
speug
spews
spewy
spial
spica
spice
spick
spics
spicy
spide
spied
spiel
spier
spies
spiff
spifs
spike
spiks
spiky
spile
spill
spilt
spims
spina
spine
spink
spins
spiny
spire
spirt
spiry
spite
spits
spitz
spivs
splat
splay
split
splog
spode
spods
spoil
spoke
spoof
spook
spool
spoom
spoon
spoor
spoot
spore
spork
sport
sposh
spots
spout
sprad
sprag
sprat
spray
spred
spree
sprew
sprig
sprit
sprod
sprog
sprue
sprug
spuds
spued
spuer
spues
spugs
spule
spume
spumy
spunk
spurn
spurs
spurt
sputa
spyal
spyre
squab
squad
squat
squaw
squeg
squib
squid
squit
squiz
stabs
stack
stade
staff
stage
stags
stagy
staid
staig
stain
stair
stake
stale
stalk
stall
stamp
stand
stane
stang
stank
staph
staps
stare
stark
starn
starr
stars
start
stash
state
stats
staun
stave
staws
stays
stead
steak
steal
steam
stean
stear
stedd
stede
steds
steed
steek
steel
steem
steen
steep
steer
steil
stein
stela
stele
stell
steme
stems
stend
steno
stens
stent
steps
stept
stere
stern
stets
stews
stewy
steys
stich
stick
stied
sties
stiff
stilb
stile
still
stilt
stime
stims
stimy
sting
stink
stint
stipa
stipe
stire
stirk
stirp
stirs
stive
stivy
stoae
stoai
stoas
stoat
stobs
stock
stoep
stogy
stoic
stoit
stoke
stole
stoln
stoma
stomp
stond
stone
stong
stonk
stonn
stony
stood
stook
stool
stoop
stoor
stope
stops
stopt
store
stork
storm
story
stoss
stots
stott
stoun
stoup
stour
stout
stove
stown
stowp
stows
strad
strae
strag
strak
strap
straw
stray
strep
strew
stria
strig
strim
strip
strop
strow
stroy
strum
strut
stubs
stuck
stude
studs
study
stuff
stull
stulm
stumm
stump
stums
stung
stunk
stuns
stunt
stupa
stupe
sture
sturt
styed
styes
style
styli
stylo
styme
stymy
styre
styte
suave
subah
subas
subby
suber
subha
succi
sucks
sucky
sucre
sudds
sudor
sudsy
suede
suent
suers
suete
suets
suety
sugan
sugar
sughs
sugos
suhur
suids
suing
suint
suite
suits
sujee
sukhs
sukuk
sulci
sulfa
sulfo
sulks
sulky
sully
sulph
sulus
sumac
sumis
summa
sumos
sumph
sumps
sunis
sunks
sunna
sunns
sunny
sunup
super
supes
supra
surah
sural
suras
surat
surds
sured
surer
sures
surfs
surfy
surge
surgy
surly
surra
sused
suses
sushi
susus
sutor
sutra
sutta
swabs
swack
swads
swage
swags
swail
swain
swale
swaly
swami
swamp
swamy
swang
swank
swans
swaps
swapt
sward
sware
swarf
swarm
swart
swash
swath
swats
swayl
sways
sweal
swear
sweat
swede
sweed
sweel
sweep
sweer
swees
sweet
sweir
swell
swelt
swept
swerf
sweys
swies
swift
swigs
swile
swill
swims
swine
swing
swink
swipe
swire
swirl
swish
swiss
swith
swits
swive
swizz
swobs
swole
swoln
swoon
swoop
swops
swopt
sword
swore
sworn
swots
swoun
swung
sybbe
sybil
syboe
sybow
sycee
syces
sycon
syens
syker
sykes
sylis
sylph
sylva
symar
synch
syncs
synds
syned
synes
synod
synth
syped
sypes
syphs
syrah
syren
syrup
sysop
sythe
syver
taals
taata
tabby
taber
tabes
tabid
tabis
tabla
table
taboo
tabor
tabun
tabus
tacan
taces
tacet
tache
tacho
tachs
tacit
tacks
tacky
tacos
tacts
taels
taffy
tafia
taggy
tagma
tahas
tahrs
taiga
taigs
taiko
tails
tains
taint
taira
taish
taits
tajes
takas
taken
taker
takes
takhi
takin
takis
takky
talak
talaq
talar
talas
talcs
talcy
talea
taler
tales
talks
talky
talls
tally
talma
talon
talpa
taluk
talus
tamal
tamed
tamer
tames
tamin
tamis
tammy
tamps
tanas
tanga
tangi
tango
tangs
tangy
tanhs
tanka
tanks
tanky
tanna
tansy
tanti
tanto
tanty
tapas
taped
tapen
taper
tapes
tapet
tapir
tapis
tappa
tapus
taras
tardo
tardy
tared
tares
targa
targe
tarns
taroc
tarok
taros
tarot
tarps
tarre
tarry
tarsi
tarts
tarty
tasar
tased
taser
tases
tasks
tassa
tasse
tasso
taste
tasty
tatar
tater
tates
taths
tatie
tatou
tatts
tatty
tatus
taube
tauld
taunt
tauon
taupe
tauts
tavah
tavas
taver
tawai
tawas
tawed
tawer
tawie
tawny
tawse
tawts
taxed
taxer
taxes
taxis
taxol
taxon
taxor
taxus
tayra
tazza
tazze
teach
teade
teads
teaed
teaks
teals
teams
tears
teary
tease
teats
teaze
techs
techy
tecta
teddy
teels
teems
teend
teene
teens
teeny
teers
teeth
teffs
teggs
tegua
tegus
tehrs
teiid
teils
teind
teins
telae
telco
teles
telex
telia
telic
tells
telly
teloi
telos
temed
temes
tempi
tempo
temps
tempt
temse
tench
tends
tendu
tenes
tenet
tenge
tenia
tenne
tenno
tenny
tenon
tenor
tense
tenth
tents
tenty
tenue
tepal
tepas
tepee
tepid
tepoy
terai
teras
terce
terek
teres
terfe
terfs
terga
terms
terne
terns
terra
terry
terse
terts
tesla
testa
teste
tests
testy
tetes
teths
tetra
tetri
teuch
teugh
tewed
tewel
tewit
texas
texes
texts
thack
thagi
thaim
thale
thali
thana
thane
thang
thank
thans
thanx
tharm
thars
thaws
thawy
thebe
theca
theed
theek
thees
theft
thegn
theic
thein
their
thelf
thema
theme
thens
theow
there
therm
these
thesp
theta
thete
thews
thewy
thick
thief
thigh
thigs
thilk
thill
thine
thing
think
thins
thiol
third
thirl
thoft
thole
tholi
thong
thorn
thoro
thorp
those
thous
thowl
thrae
thraw
three
threw
thrid
thrip
throb
throe
throw
thrum
thuds
thugs
thuja
thumb
thump
thunk
thurl
thuya
thyme
thymi
thymy
tians
tiara
tiars
tibia
tical
ticca
ticed
tices
tichy
ticks
ticky
tidal
tiddy
tided
tides
tiers
tiffs
tifos
tifts
tiger
tiges
tight
tigon
tikas
tikes
tikis
tikka
tilak
tilde
tiled
tiler
tiles
tills
tilly
tilth
tilts
timbo
timed
timer
times
timid
timon
timps
tinas
tinct
tinds
tinea
tined
tines
tinge
tings
tinks
tinny
tints
tinty
tipis
tippy
tipsy
tired
tires
tirls
tiros
tirrs
titan
titch
titer
tithe
titis
title
titre
titty
titup
tiyin
tiyns
tizes
tizzy
toads
toady
toast
toaze
tocks
tocky
tocos
today
todde
toddy
toeas
toffs
toffy
tofts
tofus
togae
togas
toged
toges
togue
tohos
toile
toils
toing
toise
toits
tokay
toked
token
toker
tokes
tokos
tolan
tolar
tolas
toled
toles
tolls
tolly
tolts
tolus
tolyl
toman
tombs
tomes
tomia
tommy
tomos
tonal
tondi
tondo
toned
toner
tones
toney
tonga
tongs
tonic
tonka
tonks
tonne
tonus
tools
tooms
toons
tooth
toots
topaz
toped
topee
topek
toper
topes
tophe
tophi
tophs
topic
topis
topoi
topos
toppy
toque
torah
toran
toras
torch
torcs
tores
toric
torii
toros
torot
torrs
torse
torsi
torsk
torso
torta
torte
torts
torus
tosas
tosed
toses
toshy
tossy
total
toted
totem
toter
totes
totty
touch
tough
touks
touns
tours
touse
tousy
touts
touze
touzy
towed
towel
tower
towie
towns
towny
towse
towsy
towts
towze
towzy
toxic
toxin
toyed
toyer
toyon
toyos
tozed
tozes
tozie
trabs
trace
track
tract
trade
trads
tragi
traik
trail
train
trait
tramp
trams
trank
tranq
trans
trant
trape
traps
trapt
trash
trass
trats
tratt
trave
trawl
trayf
trays
tread
treat
treck
treed
treen
trees
trefa
treif
treks
trema
trems
trend
tress
trest
trets
trews
treyf
treys
triac
triad
trial
tribe
trice
trick
tride
tried
trier
tries
triff
trigo
trigs
trike
trild
trill
trims
trine
trins
triol
trior
trios
tripe
trips
tripy
trist
trite
troad
troak
troat
trock
trode
trods
trogs
trois
troke
troll
tromp
trona
tronc
trone
tronk
trons
troop
trooz
trope
troth
trots
trout
trove
trows
troys
truce
truck
trued
truer
trues
trugo
trugs
trull
truly
trump
trunk
truss
trust
truth
tryer
tryke
tryma
tryps
tryst
tsade
tsadi
tsars
tsked
tsuba
tsubo
tuans
tuart
tuath
tubae
tubal
tubar
tubas
tubby
tubed
tuber
tubes
tucks
tufas
tuffe
tuffs
tufts
tufty
tugra
tuile
tuina
tuism
tuktu
tules
tulip
tulle
tulpa
tulsi
tumid
tummy
tumor
tumps
tumpy
tunas
tunds
tuned
tuner
tunes
tungs
tunic
tunny
tupek
tupik
tuple
tuque
turbo
turds
turfs
turfy
turks
turme
turms
turns
turnt
turps
turrs
tushy
tusks
tusky
tutee
tutor
tutti
tutty
tutus
tuxes
tuyer
twaes
twain
twals
twang
twank
twats
tways
tweak
tweed
tweel
tween
tweep
tweer
tweet
twerk
twerp
twice
twier
twigs
twill
twilt
twine
twink
twins
twiny
twire
twirl
twirp
twist
twite
twits
twixt
twoer
twyer
tyees
tyers
tying
tyiyn
tykes
tyler
tymps
tynde
tyned
tynes
typal
typed
types
typey
typic
typos
typps
typto
tyran
tyred
tyres
tyros
tythe
tzars
udals
udder
udons
ugali
ugged
uhlan
uhuru
ukase
ulama
ulans
ulcer
ulema
ulmin
ulnad
ulnae
ulnar
ulnas
ulpan
ultra
ulvas
ulyie
ulzie
umami
umbel
umber
umble
umbos
umbra
umbre
umiac
umiak
umiaq
ummah
ummas
ummed
umped
umphs
umpie
umpty
umrah
umras
unais
unapt
unarm
unary
unaus
unbag
unban
unbar
unbed
unbid
unbox
uncap
unces
uncia
uncle
uncos
uncoy
uncus
uncut
undam
undee
under
undid
undos
undue
undug
uneth
unfed
unfit
unfix
ungag
unget
ungod
ungot
ungum
unhat
unhip
unica
unify
union
unite
units
unity
unjam
unked
unket
unkid
unlaw
unlay
unled
unlet
unlid
unlit
unman
unmet
unmew
unmix
unpay
unpeg
unpen
unpin
unred
unrid
unrig
unrip
unsaw
unsay
unsee
unset
unsew
unsex
unsod
untax
untie
until
untin
unwed
unwet
unwit
unwon
unzip
upbow
upbye
updos
updry
upend
upjet
uplay
upled
uplit
upped
upper
upran
uprun
upsee
upset
upsey
uptak
upter
uptie
uraei
urali
uraos
urare
urari
urase
urate
urban
urbex
urbia
urdee
ureal
ureas
uredo
ureic
urena
urent
urged
urger
urges
urial
urine
urite
urman
urnal
urned
urped
ursae
ursid
urson
urubu
urvas
usage
users
usher
using
usnea
usque
usual
usure
usurp
usury
uteri
utile
utter
uveal
uveas
uvula
vacua
vaded
vades
vagal
vague
vagus
vails
vaire
vairs
vairy
vakas
vakil
vales
valet
valid
valis
valor
valse
value
valve
vamps
vampy
vanda
vaned
vanes
vangs
vants
vaped
vaper
vapes
vapid
vapor
varan
varas
vardy
varec
vares
varia
varix
varna
varus
varve
vasal
vases
vasts
vasty
vatic
vatus
vauch
vault
vaunt
vaute
vauts
vawte
vaxes
veale
veals
vealy
veena
veeps
veers
veery
vegan
vegas
veges
vegie
vegos
vehme
veils
veily
veins
veiny
velar
velds
veldt
veles
vells
velum
venae
venal
vends
vendu
veney
venge
venin
venom
vents
venue
venus
verbs
verge
verra
verry
verse
verso
verst
verts
vertu
verve
vespa
vesta
vests
vetch
vexed
vexer
vexes
vexil
vezir
vials
viand
vibes
vibex
vibey
vicar
viced
vices
vichy
video
viers
views
viewy
vifda
viffs
vigas
vigia
vigil
vigor
vilde
viler
villa
villi
vills
vimen
vinal
vinas
vinca
vined
viner
vines
vinew
vinic
vinos
vints
vinyl
viola
viold
viols
viper
viral
vired
vireo
vires
virga
virge
virid
virls
virtu
virus
visas
vised
vises
visie
visit
visne
vison
visor
vista
visto
vitae
vital
vitas
vitex
vitro
vitta
vivas
vivat
vivda
viver
vives
vivid
vixen
vizir
vizor
vleis
vlies
vlogs
voars
vocab
vocal
voces
voddy
vodka
vodou
vodun
voema
vogie
vogue
voice
voids
voila
voile
voips
volae
volar
voled
voles
volet
volks
volta
volte
volti
volts
volva
volve
vomer
vomit
voted
voter
votes
vouch
vouge
voulu
vowed
vowel
vower
voxel
vozhd
vraic
vrils
vroom
vrous
vrouw
vrows
vuggs
vuggy
vughs
vughy
vulgo
vulns
vulva
vutty
vying
waacs
wacke
wacko
wacks
wacky
wadds
waddy
waded
wader
wades
wadge
wadis
wadts
wafer
waffs
wafts
waged
wager
wages
wagga
wagon
wagyu
wahoo
waide
waifs
waift
wails
wains
wairs
waist
waite
waits
waive
wakas
waked
waken
waker
wakes
wakfs
waldo
walds
waled
waler
wales
walie
walis
walks
walla
walls
wally
walty
waltz
wamed
wames
wamus
wands
waned
wanes
waney
wangs
wanks
wanky
wanle
wanly
wanna
wants
wanty
wanze
waqfs
warbs
warby
wards
wared
wares
warez
warks
warms
warns
warps
warre
warst
warts
warty
wases
washy
wasms
wasps
waspy
waste
wasts
watap
watch
water
watts
wauff
waugh
wauks
waulk
wauls
waurs
waved
waver
waves
wavey
wawas
wawes
wawls
waxed
waxen
waxer
waxes
wayed
wazir
wazoo
weald
weals
weamb
weans
wears
weary
weave
webby
weber
wecht
wedel
wedge
wedgy
weeds
weedy
weeke
weeks
weels
weems
weens
weeny
weeps
weepy
weest
weete
weets
wefte
wefts
weids
weigh
weils
weird
weirs
weise
weize
wekas
welch
welds
welke
welks
welkt
wells
welly
welsh
welts
wembs
wench
wends
wenge
wenny
wents
weros
wersh
wests
wetas
wetly
wexed
wexes
whack
whale
whamo
whams
whang
whaps
whare
wharf
whata
whats
whaup
whaur
wheal
whear
wheat
wheel
wheen
wheep
wheft
whelk
whelm
whelp
whens
where
whets
whews
wheys
which
whids
whiff
whift
whigs
while
whilk
whims
whine
whins
whiny
whios
whips
whipt
whirl
whirr
whirs
whish
whisk
whiss
whist
white
whits
whity
whizz
whole
whomp
whoof
whoop
whoot
whops
whore
whorl
whort
whose
whoso
whows
whump
whups
whyda
wicca
wicks
wicky
widdy
widen
wider
wides
widow
width
wield
wiels
wifed
wifes
wifey
wifie
wifty
wigan
wigga
wiggy
wight
wikis
wilco
wilds
wiled
wiles
wilga
wilis
wilja
wills
willy
wilts
wimps
wimpy
wince
winch
winds
windy
wined
wines
winey
winge
wings
wingy
winks
winna
winns
winos
winze
wiped
wiper
wipes
wired
wirer
wires
wirra
wised
wiser
wises
wisha
wisht
wisps
wispy
wists
witan
witch
wited
wites
withe
withs
withy
witty
wived
wiver
wives
wizen
wizes
woads
woald
wocks
wodge
woful
wojus
woken
woker
wokka
wolds
wolfs
wolly
wolve
woman
wombs
womby
women
womyn
wonga
wongi
wonks
wonky
wonts
woods
woody
wooed
wooer
woofs
woofy
woold
wools
wooly
woons
woops
woopy
woose
woosh
wootz
woozy
words
wordy
works
world
worms
wormy
worry
worse
worst
worth
worts
would
wound
woven
wowed
wowee
woxen
wrack
wrang
wraps
wrapt
wrast
wrate
wrath
wrawl
wreak
wreck
wrens
wrest
wrick
wried
wrier
wries
wring
wrist
write
writs
wroke
wrong
wroot
wrote
wroth
wrung
wryer
wryly
wuddy
wudus
wulls
wurst
wuses
wushu
wussy
wuxia
wyled
wyles
wynds
wynns
wyted
wytes
xebec
xenia
xenic
xenon
xeric
xerox
xerus
xoana
xrays
xylan
xylem
xylic
xylol
xylyl
xysti
xysts
yaars
yabas
yabba
yabby
yacca
yacht
yacka
yacks
yaffs
yager
yages
yagis
yahoo
yaird
yakka
yakow
yales
yamen
yampy
yamun
yangs
yanks
yapok
yapon
yapps
yappy
yarak
yarco
yards
yarer
yarfa
yarks
yarns
yarrs
yarta
yarto
yates
yauds
yauld
yaups
yawed
yawey
yawls
yawns
yawny
yawps
ybore
yclad
ycled
ycond
ydrad
ydred
yeads
yeahs
yealm
yeans
yeard
yearn
years
yeast
yecch
yechs
yechy
yedes
yeeds
yeesh
yeggs
yelks
yells
yelms
yelps
yelts
yenta
yente
yerba
yerds
yerks
yeses
yesks
yests
yesty
yetis
yetts
yeuks
yeuky
yeven
yeves
yewen
yexed
yexes
yfere
yield
yiked
yikes
yills
yince
yipes
yippy
yirds
yirks
yirrs
yirth
yites
yitie
ylems
ylike
ylkes
ymolt
ympes
yobbo
yobby
yocks
yodel
yodhs
yodle
yogas
yogee
yoghs
yogic
yogin
yogis
yoick
yojan
yoked
yokel
yoker
yokes
yokul
yolks
yolky
yomim
yomps
yonic
yonis
yonks
yoofs
yoops
yores
yorks
yorps
youks
young
yourn
yours
yourt
youse
youth
yowed
yowes
yowie
yowls
yowza
yrapt
yrent
yrivd
yrneh
ysame
ytost
yuans
yucas
yucca
yucch
yucko
yucks
yucky
yufts
yugas
yuked
yukes
yukky
yukos
yulan
yules
yummo
yummy
yumps
yupon
yuppy
yurta
yurts
yuzus
zabra
zacks
zaida
zaidy
zaire
zakat
zaman
zambo
zamia
zanja
zante
zanza
zanze
zappy
zarfs
zaris
zatis
zaxes
zayin
zazen
zeals
zebec
zebra
zebub
zebus
zedas
zeins
zendo
zerda
zerks
zeros
zests
zesty
zetas
zexes
zezes
zhomo
zibet
ziffs
zigan
zilas
zilch
zilla
zills
zimbi
zimbs
zinco
zincs
zincy
zineb
zines
zings
zingy
zinke
zinky
zippo
zippy
ziram
zitis
zizel
zizit
zlote
zloty
zoaea
zobos
zobus
zocco
zoeae
zoeal
zoeas
zoism
zoist
zombi
zonae
zonal
zonda
zoned
zoner
zones
zonks
zooea
zooey
zooid
zooks
zooms
zoons
zooty
zoppa
zoppo
zoril
zoris
zorro
zouks
zowee
zowie
zulus
zupan
zupas
zuppa
zurfs
zuzim
zygal
zygon
zymes
zymic";

# ╔═╡ 013d1153-d3c0-405c-8797-3419ffeb4d4f
const dontwordle_valid_words = split(dontwordle_valid_words_raw, '\n') #these are based on the first revision of the NYT word list before it was expanded to over 14000

# ╔═╡ 65334860-bd8b-4c08-a30a-c0baf80280ce
function make_dontwordle_inds()
	dontwordle_valid_inds = BitVector(fill(false, length(nyt_valid_inds)))
	for word in dontwordle_valid_words
		i = word_index[word]
		dontwordle_valid_inds[i] = true
	end
	return dontwordle_valid_inds
end

# ╔═╡ 0cff6b40-02fa-4916-b710-98ad589b799d
const dontwordle_valid_inds = make_dontwordle_inds()

# ╔═╡ c4946f7f-5d17-4a66-a493-b2a2698dc1d5
begin
	struct DontWordleState{Nguess, Nundo}
		guesses::WordleState{Nguess}
		undos::WordleState{Nundo}
	end

	DontWordleState() = DontWordleState(WordleState(), WordleState())
	
	function Base.:(==)(s1::DontWordleState{N1, N2}, s2::DontWordleState{N1, N2}) where {N1, N2}
		(s1.guesses == s2.guesses) && (s1.undos == s2.undos)
	end

	Base.:(==)(s1::DontWordleState{0, 0}, s2::DontWordleState{0, 0}) = true

	Base.:(==)(s1::DontWordleState, s2::DontWordleState) = false

	#games are over after 6 guesses or after all undos have been used
	isterm(s::DontWordleState{6, N}) where N = true
	isterm(s::DontWordleState) = last(s.guesses.feedback_list) == 0xf2
	isterm(s::DontWordleState{0, M}) where M = false
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
#=╠═╡
const example_guess_feedback = get_feedback(example_guess, example_answer)
  ╠═╡ =#

# ╔═╡ c43af2b6-733b-4b1b-a249-bb261f059783
#=╠═╡
const example_guess_feedback_bytes = convert_bytes(example_guess_feedback)
  ╠═╡ =#

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

# ╔═╡ b75bb4fe-6b09-4d7c-8bc0-6bf224d2c95a
function get_possible_indices!(inds::BitVector, guess_list::SVector{N, UInt16}, feedback_list::SVector{N, UInt8}; baseline = wordle_original_inds) where N
	inds .= baseline
	for i in 1:N
		inds .*= get_possible_indices(guess_list[i], feedback_list[i])
	end
	return inds
end

# ╔═╡ d9c4ff04-12ba-4e1c-bc52-4c12388d514b
function calculate_state_entropy(s::WordleState; possible_indices = copy(nyt_valid_inds))
	haskey(state_entropy_lookup, s) && return state_entropy_lookup[s]
	get_possible_indices!(possible_indices, s)
	n = sum(possible_indices)
	state_entropy_lookup[s] = Float32(log2(n))
end

# ╔═╡ 1967f344-0d2e-4af0-a6b5-d8489079629d
#figure out which guess we would expectc produce the largest increase in information given a state
function find_best_information_gain_guess(possible_answer_inds::BitVector, guess_count::Integer, possible_answers::Vector{Int64}, feedback_entropies::Vector{Float32})
	l = sum(possible_answer_inds)
	iszero(l) && error("No possible answers left")

	starting_entropy = Float32(wordle_original_entropy)
	(l == 1) && return (best_guess = wordle_actions[findfirst(possible_answer_inds)], best_score = starting_entropy / (guess_count + 1), best_entropy = 0f0)
	
	view(possible_answers, 1:l) .= view(wordle_actions, possible_answer_inds)

	best_guess = 1
	best_score = 0f0
	
	best_entropy = starting_entropy

	feedback_entropies .= -1f0
	for guess_index in wordle_actions
		final_entropy = 0.0f0
		score = 0.0f0
		
		@fastmath @inbounds for i in view(possible_answers, 1:l)
			f = feedback_matrix[i, guess_index]
			if feedback_entropies[f + 0x01] == -1f0
				n = dot(get_possible_indices(guess_index, f), possible_answer_inds)
				entropy = Float32(log2(n))
				feedback_entropies[f+0x01] = entropy
			else
				entropy = feedback_entropies[f + 0x01]
			end
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
	end

	return (best_guess = best_guess, best_score = best_score, best_guess_entropy = best_entropy)
end

# ╔═╡ 5003d34f-39b1-48f0-9ee9-5d38c1c2a5f0
#figure out which guess we would expectc produce the largest increase in information given a state
function eval_guess_information_gain_scores!(scores::Vector{Float32}, entropies::Vector{Float32}, possible_answer_inds::BitVector, guess_count::Integer, possible_answers::Vector{UInt16}, feedback_entropies::Vector{Float32})
	l = sum(possible_answer_inds)
	iszero(l) && error("No possible answers left")

	# starting_entropy = Float32(wordle_original_entropy)
	starting_entropy = Float32(log2(l))
	if (l == 1) 
		# best_score = starting_entropy / (guess_count + 1)
		best_score = 1f0
		# other_score = starting_entropy / (guess_count + 2)
		other_score = 0f0
		scores .= other_score
		best_guess = findfirst(possible_answer_inds)
		scores[best_guess] = best_score
		entropies .= 0f0
		return (guess_scores = scores, guess_entropies = entropies, best_guess = best_guess, max_score = best_score, min_score = other_score)
	elseif (l == 2)
		best_score = 0.5f0
		scores .= 0f0
		best_guesses = findall(possible_answer_inds)
		scores[best_guesses] .= best_score
		entropies .= 0f0
		return (guess_scores = scores, guess_entropies = entropies, best_guess = best_guesses[1], max_score = best_score, min_score = 0f0)
	end
	
	view(possible_answers, 1:l) .= view(wordle_actions, possible_answer_inds)

	best_guess = 1
	max_score = typemin(Float32)
	min_score = typemax(Float32)
	
	best_entropy = starting_entropy

	for guess_index in wordle_actions
		final_entropy = 0.0f0
		# score = 0.0f0
		feedback_entropies .= -1f0
		win_probability = 0f0
		@fastmath @inbounds for i in view(possible_answers, 1:l)
			f = feedback_matrix[i, guess_index]
			if feedback_entropies[f + 0x01] == -1f0
				n = dot(get_possible_indices(guess_index, f), possible_answer_inds)
				entropy = Float32(log2(n))
				feedback_entropies[f+0x01] = entropy
			else
				entropy = feedback_entropies[f + 0x01]
			end
			# information_gain = starting_entropy - entropy
			win = (guess_index == i)
			win_probability += win
			# oneleft = iszero(entropy)
			# d = guess_count + win + 2*(oneleft*!win) + 3*(!oneleft)
			# score += information_gain / d
			final_entropy += entropy
		end
		
		final_entropy /= l
		win_probability /= l
		# score /= l
		score = (starting_entropy - final_entropy) / (1f0 - win_probability)
		scores[guess_index] = score
		entropies[guess_index] = final_entropy

		if score > max_score
			max_score = score
			best_guess = guess_index
		end

		if score < min_score
			min_score = score
		end
	end

	return (guess_scores = scores, guess_entropies = entropies, best_guess = best_guess, max_score = max_score, min_score = min_score)
end

# ╔═╡ 5d87942f-188a-437e-b923-7e91b9f5b923
function eval_guess_information_gain_scores(s::WordleState{N}; scores = zeros(Float32, length(nyt_valid_inds)), entropies = zeros(Float32, length(nyt_valid_inds)), answer_inds::BitVector = copy(nyt_valid_inds), possible_answers = copy(wordle_actions), feedback_entropies = zeros(Float32, 243), maxentries = Sys.total_physical_memory() / 3 / ((14855 * 64 + 16 + 32 + 32 + 64)/8)) where N
	access_time = time()
	push!(information_gain_scores.sorted_states, s)
	push!(information_gain_scores.sorted_access_times, access_time)
	if haskey(information_gain_scores.scores, s)
		output = information_gain_scores.scores[s]
		greedy_information_gain_action_lookup[s] = output.best_guess
		old_access_time = output.access_time[1]
		i = searchsortedfirst(information_gain_scores.sorted_access_times, old_access_time)
		@assert information_gain_scores.sorted_states[i] == s
		deleteat!(information_gain_scores.sorted_states, i)
		deleteat!(information_gain_scores.sorted_access_times, i)
		output.access_time[1] = access_time
		return (guess_scores = output.guess_scores, guess_entropies = output.guess_entropies, best_guess = output.best_guess, max_score = output.max_score, min_score = output.min_score)
	elseif length(information_gain_scores.scores) >= maxentries
		s_del = first(information_gain_scores.sorted_states)
		delete!(information_gain_scores.scores, s_del)
		deleteat!(information_gain_scores.sorted_states, 1)
		deleteat!(information_gain_scores.sorted_access_times, 1)
	end
	get_possible_indices!(answer_inds, s; baseline = wordle_original_inds)
	output = eval_guess_information_gain_scores!(scores, entropies, answer_inds, N, possible_answers, feedback_entropies)
	information_gain_scores.scores[s] = (;deepcopy(output)..., access_time = [access_time])
	sizehint!(information_gain_scores.scores, round(Int64, maxentries))
	greedy_information_gain_action_lookup[s] = output.best_guess
	return output
end

# ╔═╡ 9a75b05d-82fd-4a3c-8cb9-5161dfc18949
function wordle_greedy_information_gain_π(s::WordleState; kwargs...)
	haskey(greedy_information_gain_action_lookup, s) && return greedy_information_gain_action_lookup[s]
	output = eval_guess_information_gain_scores(s; kwargs...)
	greedy_information_gain_action_lookup[s] = output.best_guess
	return output.best_guess
end

# ╔═╡ 3d935fe6-16d9-4fce-8a2d-33c763801b94
function wordle_greedy_information_gain_prior!(prior::Vector{Float32}, s::WordleState; kwargs...)
	output = eval_guess_information_gain_scores(s; kwargs...)
	prior .= output.guess_scores .- output.min_score
	return Int64(output.best_guess)
end

# ╔═╡ 430ee1a8-8267-4a72-8380-e7460a28e47e
#otherwise use the normal prior
wordle_root_candidate_greedy_information_gain_prior!(args...; kwargs...) = wordle_greedy_information_gain_prior!(args...; kwargs...)

# ╔═╡ 64dbd6c6-2691-4580-97e3-bb2f875472d7
#if a state value reaches this point, then there can be no improvement
function maximum_possible_score(s::WordleState{N}; possible_indices = copy(nyt_valid_inds)) where N
	get_possible_indices!(possible_indices, s)
	l = sum(possible_indices)
	iszero(l) && error("Not a valid state")
	#from a non terminal state, the best possible outcome is to win on the next turn, this will always be the maximum score for any state and is actually achievable when there's only one answer left
	l == 1 && return -1f0 

	#if there is more than one answer left, then the best possible outcome is to either win on the next turn or guarantee a win on the following turn.  For n words left, the probability of guessing correctly on the next turn is 1/n resulting in a best possible expected value for remaining turns of (1/n) + 2*(n-1)/n = (1 + 2n - 2)/n = (2n - 1)/n
	-Float32((2*l - 1)/l)
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

# ╔═╡ 4eb18ec4-b327-4f97-a380-10469441cff8
function eval_guess_information_gain(s::WordleState{N}; possible_indices = copy(nyt_valid_inds), save_all_scores = false, kwargs...) where N
	save_all_scores && haskey(all_guesses_information_gain_lookup, s) && return all_guesses_information_gain_lookup[s]
	!save_all_scores && haskey(greedy_information_gain_lookup, s) && return greedy_information_gain_lookup[s]	
	if !save_all_scores && haskey(all_guesses_information_gain_lookup, s) 
		(guess_scores, expected_entropy, ranked_guess_inds) = all_guesses_information_gain_lookup[s]
		i_best = first(ranked_guess_inds)
		return (best_guess = i_best, best_score = guess_scores[i_best], best_guess_entropy = expected_entropy[i_best])
	end
	
	get_possible_indices!(possible_indices, s)
	out = eval_guess_information_gain(possible_indices, N; save_all_scores = save_all_scores, kwargs...)

	(length(out) == 1) && return out
	
	if save_all_scores && N < 2
		all_guesses_information_gain_lookup[s] = out
	elseif !save_all_scores
		greedy_information_gain_lookup[s] = out
	end
	
	return out
end

# ╔═╡ a383cd80-dd49-4f9f-ba67-1f58ea7eb0b6
# ╠═╡ skip_as_script = true
#=╠═╡
const wordle_start_information_gain_guesses = eval_guess_information_gain(WordleState(); save_all_scores = true) |> display_one_step_guesses
  ╠═╡ =#

# ╔═╡ 3937537e-1de7-4212-b939-0b37e315ffbf
#=╠═╡
const wordle_start_information_gain_lookup = Dict(a.word => (;a..., rank = i) for (i, a) in enumerate(wordle_start_information_gain_guesses))
  ╠═╡ =#

# ╔═╡ 35f4bf5e-0da0-4d0f-a5d8-956d773e716e
begin
	step_reward(s::WordleState{6}) = -1f0 - (s.feedback_list[6] != 0xf2)
	step_reward(s::WordleState) = -1f0
end

# ╔═╡ bd744b6d-db57-42bf-88c0-c667dfb03f5f
begin
#produces the distribution of possible feedback and thus transition states given an afterstate
function wordle_transition(s::WordleState{N}, a::UInt16; possible_indices = copy(nyt_valid_inds), possible_answers = wordle_original_inds, use_cache = true) where N
	(use_cache && haskey(wordle_transition_lookup, (s, a))) && return wordle_transition_lookup[(s, a)]
	get_possible_indices!(possible_indices, s; baseline = possible_answers)
	n = sum(possible_indices)
	rawprobabilities = SparseVector(zeros(Float32, 243))
	for answer_index in view(wordle_actions, possible_indices)
		f = feedback_matrix[answer_index, a]
		rawprobabilities[f+1] += 1f0
	end
	rawprobabilities ./= n
	transition_states = Vector{WordleState}()
	probabilities = rawprobabilities.nzval
	transition_states = Vector{WordleState}(undef, length(probabilities))
	rewards = zeros(Float32, length(probabilities))
	new_guess_list = SVector{N+1}([s.guess_list; a])
	for (i, f) in enumerate(rawprobabilities.nzind)
		s′ = WordleState(new_guess_list, SVector{N+1}([s.feedback_list; UInt8(f-1)]))
		# r = scoregame(s′)
		r = step_reward(s′)
		transition_states[i] = s′
		rewards[i] = r
	end
	output = (rewards = rewards, transition_states = transition_states, probabilities = probabilities)
	use_cache && (wordle_transition_lookup[(s, a)] = output)
	return output
end

wordle_transition(s::WordleState{N}, guess; kwargs...) where N = wordle_transition(s, conv_guess(guess); kwargs...)
end

# ╔═╡ c9b7a2bd-6f77-4df0-a50e-b71446b8a274
const wordle_transition_distribution = StateMDPTransitionDistribution(wordle_transition, WordleState())

# ╔═╡ d3d0df67-ec26-4352-9164-a53d14d1b065
const wordle_mdp = StateMDP(wordle_actions, wordle_transition_distribution, () -> WordleState(), isterm)

# ╔═╡ 05ffe4f9-8cd6-424e-8955-e3ca21c486f3
function wordle_greedy_information_gain_state_value(s::WordleState; possible_indices = copy(nyt_valid_inds))
	kwargs = make_information_gain_prior_args()
	scores = zeros(Float32, length(nyt_valid_inds))
	π(s) = wordle_greedy_information_gain_π(s; scores = scores, kwargs...)
	distribution_rollout(wordle_mdp, π, 1f0; s0 = s, usethreads=false, possible_indices = possible_indices)
end

# ╔═╡ 8cb865b4-8928-4210-96d5-6d6a71deb03b
begin
	#evaluation of starting state using greedy information gain policy
	greedy_information_gain_root_score = wordle_greedy_information_gain_state_value(WordleState())
	(expected_score = greedy_information_gain_root_score, expected_turns = score2turns(greedy_information_gain_root_score))
end

# ╔═╡ 2b8af0f4-d71e-4ff6-aad8-fd54463d587c
begin
function wordle_greedy_information_gain_guess_value(s::WordleState, guess_index::Int64; possible_indices = copy(nyt_valid_inds))
	k = (s, guess_index)
	haskey(guess_value_lookup, k) && return guess_value_lookup[k]
	kwargs = make_information_gain_prior_args()
	scores = zeros(Float32, length(nyt_valid_inds))
	π(s) = wordle_greedy_information_gain_π(s; scores = scores, kwargs...)
	v = distribution_rollout(wordle_mdp, π, 1f0; s0 = s, i_a0 = guess_index, usethreads=false, possible_indices = possible_indices)
	guess_value_lookup[k] = v
	return v
end

wordle_greedy_information_gain_guess_value(s::WordleState, guess; kwargs...) = wordle_greedy_information_gain_guess_value(s, conv_guess(guess); kwargs...)
end

# ╔═╡ 2fa84583-4ff7-48d0-bd62-cb7380535e91
function wordle_greedy_information_gain_one_step_policy_iteration(s::WordleState{N}; num_evaluations = 10) where {N}
	best_guess = 1
	best_rank = 1
	best_score = 0f0
	t0 = time()
	last_time = time()
	output = zeros(Float32, num_evaluations)

	guess_scores = eval_guess_information_gain(s; save_all_scores = true)

	isa(guess_scores, @NamedTuple{best_guess::UInt16}) && return (best_guess = guess_scores.best_guess, best_value = wordle_original_entropy/(N+1), hard_mode_indices = make_one_answer_hard_mode(guess_scores.best_guess), likely_answer_indices = make_one_answer_hard_mode(guess_scores.best_guess), sorted_guesses = [(guess = nyt_valid_words[guess_scores.best_guess], guess_index = guess_scores.best_guess, expected_value = -N-1, information_gain_rank = 1, information_gain_score = wordle_original_entropy/(N+1), expected_entropy = 0.0, answer_probability = 1.0, hard_mode_guess = true)])	
	
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
	(best_guess = best_guess, best_value = best_score, hard_mode_indices = hard_mode_indices, likely_answer_indices = likely_answer_indices, sorted_guesses = [(guess = nyt_valid_words[guess_index], guess_index = guess_index, expected_value = output[i], expected_turns = -output[i], information_gain_rank = i, information_gain_score = guess_scores.guess_scores[guess_index], expected_entropy = guess_scores.expected_entropy[guess_index], answer_probability = likely_answer_indices[guess_index] / num_likely, hard_mode_guess = Bool(hard_mode_indices[guess_index])) for (i, guess_index) in enumerate(scoreinds)][sorted_inds])
end

# ╔═╡ e804f5fc-c817-4851-a14b-fdcca1180773
#=╠═╡
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
  ╠═╡ =#

# ╔═╡ a9e89449-74ea-45d1-93eb-48d11903d03c
#=╠═╡
function display_tree_results(s::WordleState, ranked_guesses; calc_guess_value = wordle_greedy_information_gain_guess_value)
	information_gain_results = eval_guess_information_gain_scores(s)

	ranked_guess_inds = sortperm(information_gain_results.guess_scores; rev=true)
	
	word_rank_lookup = Dict(nyt_valid_words[ranked_guess_inds[i]] => i for i in eachindex(nyt_valid_words))

	hard_mode_indices = get_possible_indices(s; baseline = nyt_valid_inds)
	likely_answer_indices = get_possible_indices(s)
	num_hard_mode = sum(hard_mode_indices)
	num_likely = sum(likely_answer_indices)

	hard_mode_words = nyt_valid_words[hard_mode_indices]
	answer_words = nyt_valid_words[likely_answer_indices]
	
	score_table = [begin
		wordind = word_index[a.word]
		policy_value = calc_guess_value(s, a.word; possible_indices = test_possible_indices)
		tree_improvement = a.values - policy_value
		(word = a.word, expected_turns = -a.values, visits = round(Int64, a.visits), tree_value = a.values, policy_value = policy_value, policy_rank = word_rank_lookup[a.word], tree_improvement = tree_improvement < 1f-4 ? 0.0 : round(tree_improvement; sigdigits = 2)) 
	end
	for a in ranked_guesses] |> DataFrame

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
  ╠═╡ =#

# ╔═╡ 6fdc99b1-beea-4c45-98bb-d257501a6878
#=╠═╡
function show_wordle_mcts_guesses(visit_counts::Dict, values::Dict, s::WordleState; kwargs...)
	ranked_guesses = sort([(word = nyt_valid_words[i], visits = visit_counts[s][i], values = values[s][i]) for i in visit_counts[s].nzind]; by = t -> t.values, rev=true)
	display_tree_results(s, ranked_guesses; kwargs...)
end
  ╠═╡ =#

# ╔═╡ c8d9a991-ff7b-448e-820c-a1f4a89ee22e
#=╠═╡
isnothing(root_candidate_mcts_options) ? md"""Waiting for results""" : display_tree_results(WordleState(), root_candidate_mcts_options)
  ╠═╡ =#

# ╔═╡ 09c89ae6-83e2-4f86-8c3f-b1528235c70a
#=╠═╡
#expected score for random game
begin
	random_game_score = mean(sample_rollout(wordle_mdp, s -> rand(wordle_actions), 1f0) for _ in 1:1_000)
	(expected_score = random_game_score, expected_turns = score2turns(random_game_score))
end
  ╠═╡ =#

# ╔═╡ e093dd57-040b-4983-a95f-057097fcefff
function run_wordle_mcts(s::WordleState, nsims::Integer; π_dist! = wordle_greedy_information_gain_prior!, prior_kwargs = make_information_gain_prior_args(), topn = 10, p_scale = 100f0, kwargs...)
	possible_indices = copy(nyt_valid_inds)
	max_value(s) = maximum_possible_score(s; possible_indices = possible_indices)
	(mcts_guess, visit_counts, values) = monte_carlo_tree_search(wordle_mdp, 1f0, s, (prior, s) -> π_dist!(prior, s; prior_kwargs...), p_scale, topn; nsims = nsims, sim_message=true, compute_max_value = max_value, kwargs...)
	return (visit_counts, values)
end

# ╔═╡ e92e232e-9e8c-4b84-aa1b-49a67a079380
# ╠═╡ show_logs = false
const (root_wordle_visit_counts, root_wordle_values) = run_wordle_mcts(WordleState(), 1; 
		topn = 10, 
		p_scale = 100f0,
		c = 1f0,
		make_step_kwargs = k -> (possible_indices = test_possible_indices,))

# ╔═╡ 58ce6598-0cf0-4450-b50f-afc2da287755
function wordle_root_tree_policy(s::WordleState)
	values = root_wordle_values[s]
	visits = root_wordle_visit_counts[s]
	i = argmax(values[i] for i in visits.nzind)
	visits.nzind[i]
end

# ╔═╡ 164fccfc-56c5-4538-acf1-90ec13db38f8
Base.summarysize(root_wordle_values)

# ╔═╡ a41be793-431d-42f0-885b-7ff89efa0252
distribution_rollout(wordle_mdp, wordle_greedy_information_gain_π, 1f0) |> score2turns

# ╔═╡ f6253866-7bef-400f-9713-e0fd5054b201
#games will be scored at a maximum of 6 for winning on the first guess down to 0 for failing to win within 6 guesses
begin
	scoregame(s::WordleState{0}) = 0f0
	function scoregame(s::WordleState{N}) where N
		win = Float32(s.feedback_list[N] == 0xf2)
		(6f0 - N + 1f0)*win
	end
end

# ╔═╡ 8ab11beb-930a-43cb-9a52-f19f49819f1b
function run_wordle_answer_game(s::WordleState, π::Function, true_answer_index::Integer; kwargs...)
	i_a = π(s; kwargs...)
	while !isterm(s)
		f = feedback_matrix[true_answer_index, i_a]
		s = WordleState(vcat(s.guess_list, wordle_actions[i_a]), vcat(s.feedback_list, f))
		i_a = π(s; kwargs...)
	end
	return s
end

# ╔═╡ 7a9e09f2-70ca-4180-9c1c-56d74e743098
function run_wordle_answer_games(s0::WordleState, π::Function; possible_indices = copy(nyt_valid_inds))
	get_possible_indices!(possible_indices, s0; baseline = wordle_original_inds)
	answer_indices = findall(possible_indices)
	games = Vector{WordleState}(undef, length(answer_indices))
	for (i, answer_index) in enumerate(answer_indices)
		s = s0
		while !isterm(s)
			i_a = π(s)
			f = feedback_matrix[answer_index, i_a]
			s = WordleState(vcat(s.guess_list, wordle_actions[i_a]), vcat(s.feedback_list, f))
		end
		games[i] = deepcopy(s)
	end
	return games, answer_indices
end

# ╔═╡ b301b451-3276-45fa-8777-eb2069b3e580
function analyze_wordle_policy_over_answers(s::WordleState, π::Function; kwargs...)
	games, answer_indices = run_wordle_answer_games(s, π; kwargs...)
	game_length_words = [Vector{String}() for _ in 1:6]
	for (i, g) in enumerate(games)
		@assert g.feedback_list[end] == 0xf2 #make sure that every game is a win
		push!(game_length_words[game_length(g)], nyt_valid_words[answer_indices[i]])
	end
	return game_length_words
end

# ╔═╡ c58f7788-5d64-45af-b5a1-9b30c24f730c
function compare_wordle_polices_over_answers(s::WordleState, π1::Function, π2::Function)
	games1, answer_indices1 = run_wordle_answer_games(s, π1)
	games2, answer_indices2 = run_wordle_answer_games(s, π2)
	@assert answer_indices1 == answer_indices2
	result_compare = Dict((l1, l2) => Set{String}() for l1 in 1:6 for l2 in 1:6)
	for i in eachindex(answer_indices1)
		l1 = game_length(games1[i])
		l2 = game_length(games2[i])
		push!(result_compare[(l1, l2)], nyt_valid_words[answer_indices1[i]])
	end
	return result_compare
end

# ╔═╡ 71e35ad3-1c42-4ffd-946b-5eb9e6b72f86
begin
	#checks to see if a wordle state is comptabible with another so that means that the guesses and feedback of one state could be a later version of the other state
	check_state_consistency(s::WordleState{0}, s_check::WordleState{0}) = true
	check_state_consistency(s::WordleState{0}, s_check::WordleState) = true
	check_state_consistency(s::WordleState, s_check::WordleState{0}) = false
	check_state_consistency(s::WordleState{N}, s_check::WordleState{N}) where N = s == s_check
	function check_state_consistency(s::WordleState{N1}, s_check::WordleState{N2}) where {N1, N2}
		N2 < N1 && return false
		s == get_wordle_substate(s_check, N1)
	end
end

# ╔═╡ b88f9cc7-6232-41b3-92b3-3e920d446a5f
#figure out which guess we would expectc produce the largest increase in information given a state
function eval_hardmode_guess_information_gain_scores!(scores::Vector{Float32}, entropies::Vector{Float32}, allowed_guess_inds::BitVector, possible_answer_inds::BitVector, guess_count::Integer, possible_answers::Vector{UInt16}, feedback_entropies::Vector{Float32})
	l = sum(possible_answer_inds)
	iszero(l) && error("No possible answers left")

	# starting_entropy = Float32(wordle_original_entropy)
	starting_entropy = Float32(log2(l))
	scores .= 0f0
	entropies .= 0f0
	if (l == 1) 
		# best_score = starting_entropy / (guess_count + 1)
		best_score = 1f0
		# other_score = starting_entropy / (guess_count + 2)
		best_guess = findfirst(possible_answer_inds)
		scores[best_guess] = best_score
		return (guess_scores = scores, guess_entropies = entropies, best_guess = best_guess, max_score = best_score, min_score = 0f0)
	elseif (l == 2)
		best_score = 0.5f0
		best_guesses = findall(possible_answer_inds)
		scores[best_guesses] .= best_score
		return (guess_scores = scores, guess_entropies = entropies, best_guess = best_guesses[1], max_score = best_score, min_score = 0f0)
	end
	
	view(possible_answers, 1:l) .= view(wordle_actions, possible_answer_inds)

	best_guess = 1
	max_score = typemin(Float32)
	min_score = typemax(Float32)
	
	best_entropy = starting_entropy

	for guess_index in view(wordle_actions, allowed_guess_inds)
		final_entropy = 0.0f0
		# score = 0.0f0
		feedback_entropies .= -1f0
		win_probability = 0f0
		@fastmath @inbounds for i in view(possible_answers, 1:l)
			f = feedback_matrix[i, guess_index]
			if feedback_entropies[f + 0x01] == -1f0
				n = dot(get_possible_indices(guess_index, f), possible_answer_inds)
				entropy = Float32(log2(n))
				feedback_entropies[f+0x01] = entropy
			else
				entropy = feedback_entropies[f + 0x01]
			end
			# information_gain = starting_entropy - entropy
			win = (guess_index == i)
			win_probability += Float32(win)
			# oneleft = iszero(entropy)
			# d = guess_count + win + 2*(oneleft*!win) + 3*(!oneleft)
			# score += information_gain / d
			final_entropy += entropy
		end
		
		final_entropy /= l
		win_probability /= l
		# score /= l
		score = (starting_entropy - final_entropy) / (1f0 - win_probability)
		scores[guess_index] = score
		entropies[guess_index] = final_entropy

		if score > max_score
			max_score = score
			best_guess = guess_index
		end

		if score < min_score
			min_score = score
		end
	end

	return (guess_scores = scores, guess_entropies = entropies, best_guess = best_guess, max_score = max_score, min_score = min_score)
end

# ╔═╡ 839c4998-ac37-4964-8590-11d4840c83aa
function eval_hardmode_guess_information_gain_scores(s::WordleState{N}; scores = zeros(Float32, length(nyt_valid_inds)), entropies = zeros(Float32, length(nyt_valid_inds)), allowed_guess_inds = copy(nyt_valid_inds), possible_answer_inds = copy(nyt_valid_inds), possible_answers = copy(wordle_actions), feedback_entropies = zeros(Float32, 243)) where N
	haskey(hardmode_scores, s) && return hardmode_scores[s]
	get_possible_indices!(allowed_guess_inds, s; baseline = nyt_valid_inds)
	possible_answer_inds .= allowed_guess_inds .* wordle_original_inds
	output = eval_hardmode_guess_information_gain_scores!(scores, entropies, allowed_guess_inds, possible_answer_inds, N, possible_answers, feedback_entropies)
	hardmode_scores[s] = (guess_scores = SparseVector(output.guess_scores), guess_entropies = SparseVector(output.guess_entropies), best_guess = output.best_guess, max_score = output.max_score, min_score = output.min_score)
	hardmode_greedy_lookup[s] = Int64(output.best_guess)
	return hardmode_scores[s]
end

# ╔═╡ 6670e6d3-ff00-4a91-98c2-830c3aed92de
function wordle_hardmode_greedy_information_gain_π(s::WordleState; kwargs...)
	haskey(hardmode_greedy_lookup, s) && return hardmode_greedy_lookup[s]
	output = eval_hardmode_guess_information_gain_scores(s; kwargs...)
	hardmode_greedy_lookup[s] = Int64(output.best_guess)
end

# ╔═╡ 460d6485-b13e-4deb-a0bd-53b9ebe0c47f
distribution_rollout(wordle_mdp, s -> wordle_hardmode_greedy_information_gain_π(s), 1f0) |> score2turns

# ╔═╡ 5df48271-77d3-4ab5-aba2-bd3cc0b9f078
begin
function wordle_hardmode_greedy_information_gain_guess_value(s::WordleState, guess_index::Int64; possible_indices = copy(nyt_valid_inds), possible_answers = copy(wordle_actions))
	kwargs = make_hardmode_information_gain_kwargs()
	π(s) = wordle_hardmode_greedy_information_gain_π(s; kwargs...)
	distribution_rollout(wordle_mdp, π, 1f0; s0 = s, i_a0 = guess_index, usethreads=false, possible_indices = possible_indices)
end

wordle_hardmode_greedy_information_gain_guess_value(s::WordleState, guess; kwargs...) = wordle_hardmode_greedy_information_gain_guess_value(s, conv_guess(guess); kwargs...)
end

# ╔═╡ 4dd064bc-db44-4b18-90ed-5bc67c12774e
function wordle_hardmode_greedy_information_gain_prior!(prior::Vector{Float32}, s::WordleState; kwargs...)
	output = eval_hardmode_guess_information_gain_scores(s; kwargs...)
	prior .= 0f0
	if output.max_score == output.min_score
		prior[output.guess_scores.nzind] .= 1f0
	else
		priorsum = 0f0
		@inbounds @simd for i in eachindex(output.guess_scores.nzind)
			v = output.guess_scores.nzval[i] - output.min_score
			prior[output.guess_scores.nzind[i]] = output.guess_scores.nzval[i] - output.min_score
			priorsum += v
		end
		@assert priorsum > 0 "For state $s the prior distribution is all 0"
	end
	return Int64(output.best_guess)
end

# ╔═╡ 6c514cb4-03d9-4108-aca3-489212bea90e
const (root_wordle_hardmode_visit_counts, root_wordle_hardmode_values) = run_wordle_mcts(WordleState(), 1; 
		topn = 10, 
		p_scale = 100f0,
		c = 1f0,
		make_step_kwargs = k -> (possible_indices = test_possible_indices,),
		π_dist! = wordle_hardmode_greedy_information_gain_prior!,
		prior_kwargs = make_hardmode_information_gain_kwargs())

# ╔═╡ d5a9ab96-beb1-4f72-b049-2795ee0c5940
const wordle_hardmode_mdp = StateMDP(wordle_actions, wordle_transition_distribution, () -> WordleState(), isterm; is_valid_action = (s, i_a; kwargs...) -> get_possible_indices(s; baseline = nyt_valid_inds, kwargs...)[i_a])

# ╔═╡ ed8472be-d0e2-4e9d-b752-fd64250e1db2
const dontwordle_actions = [wordle_actions; UInt16(0)] #the action of 0 here represents undo

# ╔═╡ 8895b0b9-9fd1-4f5d-8132-564d9fecbf3d
begin
	#every step will be a reward of 0 except for a win.  a win can only occur if 6 guesses are made and the final guess is not the correct word
	scoregame(s::DontWordleState{6, N}) where N = Float32(s.guesses.feedback_list[6] != 0xf2)
	scoregame(s::DontWordleState) = 0f0
end

# ╔═╡ 50135114-e4f8-4042-805a-05425318f55e
begin
	conv_dontwordle_action(a::Integer) = UInt16(a)
	conv_dontwordle_action(word::AbstractString) = conv_guess(word)
	conv_dontwordle_action(s::Symbol) = s == :undo ? UInt16(0) : error("not a valid action")
end

# ╔═╡ e1252197-4592-4485-8390-40253c01f6b6
const dontwordle_transition_lookup = Dict{Tuple{DontWordleState, UInt16}, @NamedTuple{rewards::Vector{Float32}, transition_states::Vector{DontWordleState}, probabilities::Vector{Float32}}}()

# ╔═╡ b0b4164d-7299-4e5d-a993-01ffafefa43b
begin
#produces the distribution of possible feedback and thus transition states given an afterstate
function dontwordle_transition(s::DontWordleState{Nguess, Nundo}, a::UInt16, maxundos::Integer; possible_indices = copy(nyt_valid_inds), possible_answers = wordle_original_inds, usecache = true) where {Nguess, Nundo}
	isterm(s) && error("Cannot transition out of terminal state")
	usecache && haskey(dontwordle_transition_lookup, (s, a)) && return dontwordle_transition_lookup[(s, a)]
	a == 0 && Nguess == 0 && error("No guess to undo")
	a == 0 && Nundo == maxundos && error("No more undos left")
	a == 0 && return (rewards = [0f0], transition_states = [DontWordleState(WordleState(s.guesses.guess_list[1:Nguess-1], s.guesses.feedback_list[1:Nguess-1]), WordleState([s.undos.guess_list; s.guesses.guess_list[Nguess]], [s.undos.feedback_list; s.guesses.feedback_list[Nguess]]))], probabilities = [1f0])
	
	get_possible_indices!(possible_indices, s.guesses; baseline = dontwordle_valid_inds)
	
	#check to see that guess is a valid hardmode guess
	!possible_indices[a] && error("the guess $(nyt_valid_words[a]) is not one of the remaining answers for state $s") 

	possible_indices .*= possible_answers
	#check which answers are still possible including undos
	get_possible_indices!(possible_indices, s.undos; baseline = possible_indices)
	
	n = sum(possible_indices)
	rawprobabilities = SparseVector(zeros(Float32, 243))
	
	for answer_index in view(wordle_actions, possible_indices)
		f = feedback_matrix[answer_index, a]
		rawprobabilities[f+1] += 1f0
	end
	rawprobabilities ./= n
	probabilities = rawprobabilities.nzval
	transition_states = Vector{DontWordleState}(undef, length(probabilities))
	rewards = zeros(Float32, length(probabilities))
	new_guess_list = SVector{Nguess+1}([s.guesses.guess_list; a])
	for (i, f) in enumerate(rawprobabilities.nzind)
		wordle_s′ = WordleState(new_guess_list, SVector{Nguess+1}([s.guesses.feedback_list; UInt8(f-1)]))
		s′ = DontWordleState(wordle_s′, s.undos)
		r = scoregame(s′)
		transition_states[i] = s′
		rewards[i] = r
	end
	output = (rewards = rewards, transition_states = transition_states, probabilities = probabilities)
	if usecache
		dontwordle_transition_lookup[(s, a)] = output
	end
	return output
end

dontwordle_transition(s::DontWordleState, action, maxundos::Integer; kwargs...) = dontwordle_transition(s, conv_dontwordle_action(action))
end

# ╔═╡ c52535b6-cfbe-4207-8d7d-1787cb58c3ca
function create_dontwordle_mdp(maxundos::Integer)
	possible_indices = copy(nyt_valid_inds)
	step(s::DontWordleState, i_a::Integer; kwargs...) = dontwordle_transition(s, dontwordle_actions[i_a], maxundos; kwargs...)
	transition = StateMDPTransitionDistribution(step, DontWordleState())
	StateMDP(dontwordle_actions, transition, () -> DontWordleState(), isterm)
end

# ╔═╡ 43ccabd2-3688-4016-88d7-29a26d46986c
const dontwordle_mdp = create_dontwordle_mdp(2)

# ╔═╡ 9a8b5ba7-45c0-4190-b047-5e9c9e10c530
#figure out which guess we would expectc produce the largest increase in information given a state
function eval_guess_dontwordle_score(possible_answer_inds::BitVector, valid_guess_inds::BitVector, undos_left::Integer; possible_answers = copy(wordle_actions), save_all_scores = false)
	#number of words left that could be the true answer
	l = sum(possible_answer_inds)
	iszero(l) && error("No possible answers left")

	#number of valid hard mode guesses left
	lhard = sum(valid_guess_inds)

	#usually the number of hard mode guesses is larger than the number of possible answers, that way we can safely make a guess that is guaranteed not to wordle.  If that is not the case meaning that there are no hard mode guesses left that couldn't also be real answers, then we should use an undo if it is available.
	(undos_left > 0) && (lhard == l) && return (best_action = length(dontwordle_actions),)

	#if only one hardmode guess is left then select that word
	(lhard == 1) && return (best_action = findfirst(valid_guess_inds),)
	
	view(possible_answers, 1:l) .= view(wordle_actions, possible_answer_inds)

	if save_all_scores
		expected_entropy = zeros(Float32, lhard)
	end

	best_guess = findfirst(valid_guess_inds)
	highest_entropy = 0f0
	
	if l > 1000
		feedback_entropies = zeros(Float32, 243)
		#iterate through all of the possible answers but calculate the entropy based on the number of hard mode guesses left
		for (j, guess_index) in enumerate(view(wordle_actions, valid_guess_inds))
			final_entropy = 0.0f0
			# @fastmath @inbounds @simd for i in view(possible_answers, 1:l)
			@fastmath @inbounds for f in 0x00:0xf2
				n = dot(get_possible_indices(guess_index, f), valid_guess_inds)
				feedback_entropies[f+1] = Float32(log2(n))
			end
			
			@fastmath @inbounds @simd for i in view(possible_answers, 1:l)
				f = feedback_matrix[i, guess_index]
				final_entropy += feedback_entropies[f+1]
			end
	
			final_entropy /= l
			
			if final_entropy > highest_entropy
				highest_entropy = final_entropy
				best_guess = guess_index
			end
	
			if save_all_scores
				expected_entropy[j] = final_entropy
			end
		end
	else
		for (j, guess_index) in enumerate(view(wordle_actions, valid_guess_inds))
			final_entropy = 0.0f0
			@fastmath @inbounds for i in 1:l
				answer_index = possible_answers[i]
				f = feedback_matrix[answer_index, guess_index]
				n = dot(get_possible_indices(guess_index, f), valid_guess_inds)
				final_entropy += Float32(log2(n))
			end
	
			final_entropy /= l
			if final_entropy > highest_entropy
				highest_entropy = final_entropy
				best_guess = guess_index
			end
	
			if save_all_scores
				expected_entropy[j] = final_entropy
			end
		end
	end

	#every word leaves only one valid guess left
	best_guess_out = if highest_entropy == 0f0
		if lhard == l
			(best_action = best_guess, best_guess_entropy = 0f0)
		else
			best_guess = findfirst(i -> valid_guess_inds[i] && !possible_answer_inds[i], eachindex(valid_guess_inds))
			(best_action = best_guess, best_guess_entropy = 0f0)
		end
	else
		(best_action = best_guess, best_guess_entropy = highest_entropy)
	end

	!save_all_scores && return best_guess_out
	
	(valid_guesses = findall(valid_guess_inds), expected_entropy = expected_entropy, ranked_guess_inds = sortperm(expected_entropy, rev=true))
end

# ╔═╡ e7b49bac-0785-4311-a2e9-fa618bcf5eb7
begin
	const dontwordle_score_lookup = Dict{Tuple{DontWordleState, Integer}, NamedTuple}()
	const dontwordle_full_score_lookup = Dict{Tuple{DontWordleState, Integer}, NamedTuple}()
end

# ╔═╡ db05290f-ae09-4321-9979-dda7ddb3bb18
function eval_guess_dontwordle_score(s::DontWordleState{Nguess, Nundo}, maxundos::Integer; possible_indices = copy(nyt_valid_inds), hardmode_indices = copy(nyt_valid_inds), save_all_scores = false, kwargs...) where {Nguess, Nundo}
	save_all_scores && haskey(dontwordle_full_score_lookup, (s, maxundos)) && return dontwordle_full_score_lookup[(s, maxundos)]
	!save_all_scores && haskey(dontwordle_score_lookup, (s, maxundos)) && return dontwordle_score_lookup[(s, maxundos)]
	if !save_all_scores && haskey(dontwordle_full_score_lookup, (s, maxundos))
		output = dontwordle_full_score_lookup[(s, maxundos)]
		length(output) == 1 && return output
		best_guess_index = output.ranked_guess_inds[1]
		return (best_action = output.valid_guesses[best_guess_index], best_guess_entropy = output.expected_entropy[best_guess_index])
	end
	get_possible_indices!(hardmode_indices, s.guesses; baseline = dontwordle_valid_inds)

	get_possible_indices!(possible_indices, s.undos; baseline = hardmode_indices)
	possible_indices .*= wordle_original_inds

	undos_left = maxundos - Nundo
	# undos_left <=1 && @info "In state $s only have $undos_left undos left"
	output = eval_guess_dontwordle_score(possible_indices, hardmode_indices, undos_left; save_all_scores = save_all_scores, kwargs...)
	if save_all_scores
		dontwordle_full_score_lookup[(s, maxundos)] = output
	else
		dontwordle_score_lookup[(s, maxundos)] = output
	end
	return output
end

# ╔═╡ 70b62b2d-c833-4b58-9f86-81980d3dd06a
function dontwordle_greedy_entropy_π(s::DontWordleState, maxundos::Integer; kwargs...)
	score = eval_guess_dontwordle_score(s, maxundos; kwargs...)
	score.best_action
end

# ╔═╡ f4631d73-9465-479d-828d-6924d03a6cda
function dontwordle_greedy_entropy_state_value(s::DontWordleState, maxundos::Integer; possible_indices = copy(nyt_valid_inds), possible_answers = copy(wordle_actions), hardmode_indices = copy(nyt_valid_inds))
	π(s) = dontwordle_greedy_entropy_π(s, maxundos; possible_indices = possible_indices, hardmode_indices = hardmode_indices, possible_answers = possible_answers)
	distribution_rollout(dontwordle_mdp, π, 1f0; s0 = s, usethreads=false, possible_indices = possible_indices)
end

# ╔═╡ 3f73964c-c171-4759-888e-af43e53b4b2e
function dontwordle_greedy_entropy_state_guess_value(s::DontWordleState, action_index::Int64, maxundos::Integer; possible_indices = copy(nyt_valid_inds), possible_answers = copy(wordle_actions), hardmode_indices = copy(nyt_valid_inds))
	π(s) = dontwordle_greedy_entropy_π(s, maxundos; possible_indices = possible_indices, hardmode_indices = hardmode_indices, possible_answers = possible_answers)
	distribution_rollout(dontwordle_mdp, π, 1f0; s0 = s, i_a0 = action_index, usethreads=false, possible_indices = possible_indices)
end

# ╔═╡ 3424be60-3f82-450e-9293-7b1f58542f7e
const root_dontwordle_scores = eval_guess_dontwordle_score(DontWordleState(), 2; save_all_scores=true)

# ╔═╡ 66baf9cc-a76c-429d-b143-f46e82134602
const top_scoring_root_dontwordle_guesses = [(word = nyt_valid_words[i], index = i) for i in root_dontwordle_scores.ranked_guess_inds]

# ╔═╡ 3b765ff5-5d48-4525-9e3b-d374815eb4c4
# ╠═╡ show_logs = false
# ╠═╡ disabled = true
#=╠═╡
const dontwordle_policy_iteration = [begin
	guess_index = root_dontwordle_scores.ranked_guess_inds[i]
	v = dontwordle_greedy_entropy_state_guess_value(DontWordleState(), guess_index, 5) 
	@info "Done with guess rank $i = $(nyt_valid_words[guess_index]) with an expected value of $v"
	(guess = nyt_valid_words[guess_index], expected_value = v, greedy_policy_rank = i, expected_entropy = root_dontwordle_scores.expected_entropy[guess_index])
end
for i in 1:10] |> DataFrame |> d -> sort(d, :expected_value, rev=true)
  ╠═╡ =#

# ╔═╡ e3a0438a-24cf-498e-bb38-362e678e0992
const test_hardmode_indices = copy(nyt_valid_inds)

# ╔═╡ 0cb3c137-99f4-4ec7-95c4-1ec3e1322d3d
function dontwordle_greedy_entropy_prior!(prior::Vector{Float32}, s::DontWordleState{Nguess, Nundo}, maxundos::Integer; kwargs...) where {Nguess, Nundo}
	output = eval_guess_dontwordle_score(s, maxundos; save_all_scores = true, possible_indices = test_possible_indices, hardmode_indices = test_hardmode_indices)

	if haskey(dontwordle_full_score_lookup, (s, maxundos)) #in this case the possible indices won't be updated
		#hardmode guesses
		get_possible_indices!(test_hardmode_indices, s.guesses; baseline = dontwordle_valid_inds)
		
		#actual answer guesses
		get_possible_indices!(test_possible_indices, s.undos; baseline = test_hardmode_indices)
		test_possible_indices .*= wordle_original_inds
	end

	undos_left = maxundos - Nundo
	allow_undo = (undos_left > 0) && (Nguess > 0)
	
	if length(output) == 1
		# @info "in state $s there is only one likely action $(output[1])"
		prior .= 0f0
		prior[output[1]] = 1f0
		return output[1]
	else
		valid_guesses = output.valid_guesses
		min_score = output.expected_entropy[last(output.ranked_guess_inds)]
		max_score = output.expected_entropy[first(output.ranked_guess_inds)]
		
		if max_score == min_score #in this case all of the words likely only leave one valid answer
			num_answer = sum(test_possible_indices)
			num_guess = sum(test_hardmode_indices)
			prior[end] = Float32(allow_undo)
			if num_answer == num_guess
				# @info "hard mode words: $(nyt_valid_words[test_hardmode_indices])"
				view(prior, 1:(length(prior) - 1)) .= test_hardmode_indices #select from among the valid hard mode guesses
				allow_undo && return length(dontwordle_actions)
			else
				# @info "hard mode words: $(nyt_valid_words[test_hardmode_indices])"
				view(prior, 1:(length(prior) - 1)) .= (test_hardmode_indices .- test_possible_indices) #select from among the valid hard mode guesses that are also not answers
			end
			return findfirst(isone, prior)
		else
			guess_scores = output.expected_entropy
			prior .= 0f0
			prior[end] = max_score*Float32(allow_undo)
			view(prior, valid_guesses) .= guess_scores .- min_score
			return valid_guesses[first(output.ranked_guess_inds)]
		end
	end
end

# ╔═╡ af66fab5-3921-4fdf-9820-ca6ad5a21769
function run_dontwordle_mcts(s::DontWordleState, maxundos::Integer, nsims::Integer; π_dist! = (prior, s; kwargs...) -> dontwordle_greedy_entropy_prior!(prior, s, maxundos; kwargs...), topn = 10, p_scale = 100f0, mdp = dontwordle_mdp, kwargs...)
	possible_indices = copy(nyt_valid_inds)
	possible_answers = copy(wordle_actions)
	(mcts_guess, visit_counts, values) = monte_carlo_tree_search(mdp, 1f0, s, π_dist!, p_scale, topn; nsims = nsims, compute_max_value = s -> 1f0, sim_message=true, kwargs...)
	return (visit_counts, values)
end

# ╔═╡ 773f6fa9-8b07-40ad-9c01-cf1a27ceeca8
#=╠═╡
md"""
### Don'twordle MCTS
"""
  ╠═╡ =#

# ╔═╡ 2a125dec-d4bb-4214-bec8-db95821a6d45
const (dontwordle_root_visits, dontwordle_root_values) = run_dontwordle_mcts(DontWordleState(), 2, 1)

# ╔═╡ 1919ceb3-31aa-49f0-929d-3faa034e2f63
function show_dontwordle_mcts_guesses(visit_counts::Dict, values::Dict, s::DontWordleState; maxundos = 2, kwargs...)
	if !haskey(visit_counts, s)
		(visit_counts, values) = run_dontwordle_mcts(DontWordleState(), maxundos, 100)
	end
	inds = visit_counts[s].nzind
	scores = eval_guess_dontwordle_score(s, maxundos; save_all_scores=true)
	length(scores) == 1 && return [nyt_valid_words; "undo action"][scores.best_action]
	valid_guesses = scores.valid_guesses
	policy_rank_lookup = Dict(valid_guesses[scores.ranked_guess_inds[i]] => i for i in eachindex(scores.ranked_guess_inds))
	undoind = length(dontwordle_actions)
	ranked_guesses = sort([(word = i == undoind ? "undo action" : nyt_valid_words[i], visits = round(Int64, visit_counts[s][i]), values = values[s][i], policy_rank = i == undoind ? nothing : policy_rank_lookup[i]) for i in inds]; by = t -> t.values, rev=true)
	DataFrame(ranked_guesses)
end

# ╔═╡ b28792a8-319a-4b16-9f8d-eec2c4b4be49
function dontwordle_greedy_excludeguess_prior!(prior::Vector{Float32}, s::DontWordleState; exclude_guess_indices = [12221, 12472],maxundos = 5, kwargs...)
	action = dontwordle_greedy_entropy_prior!(prior, s, maxundos; kwargs...)
	prior[exclude_guess_indices] .= 0f0
	if iszero(sum(prior))
		inds = get_possible_indices(s.guesses; baseline = nyt_valid_inds)
		view(prior, 1:length(nyt_valid_inds)) .= inds
		prior[exclude_guess_indices] .= 0f0
	end
	return argmax(prior)
end

# ╔═╡ 5c40b6c3-3638-4c26-9e74-235a6cb90078
#=╠═╡
md"""
### Don't Wordle New Test State
"""
  ╠═╡ =#

# ╔═╡ bb131186-d25c-465a-9c55-46dc2a2f0256
#=╠═╡
md"""
### One Undo Variation
"""
  ╠═╡ =#

# ╔═╡ 1817dd59-2cbd-4704-88dd-e3ec761c9163
const dontwordle_oneundo_mdp = create_dontwordle_mdp(1)

# ╔═╡ 98ea8bac-11a6-4e52-a197-47b190a306a6
const (dontwordle_oneundo_root_visits, dontwordle_oneundo_root_values) = run_dontwordle_mcts(DontWordleState(), 1, 1; mdp = dontwordle_oneundo_mdp)

# ╔═╡ 955df8dd-a52b-4df9-a941-799a9e7d2d27
#=╠═╡
md"""
### No Undo Variation
"""
  ╠═╡ =#

# ╔═╡ a7dfa2f2-724a-40c0-94dc-0aeeba6c40ac
const dontwordle_noundo_mdp = create_dontwordle_mdp(0)

# ╔═╡ 510e77ba-ac46-475d-a7e1-a734e31f4bcc
const (dontwordle_noundo_root_visits, dontwordle_noundo_root_values) = run_dontwordle_mcts(DontWordleState(), 0, 1; mdp = dontwordle_noundo_mdp, c = 1f0)

# ╔═╡ 17daab93-b3ad-4eea-a032-05e61b2fb690
#=╠═╡
md"""
## Absurdle Environment

Absurdle is a deterministic version of wordle where the feedback for a guess is always the least informative with respect to the list of possible answers.  When the game starts, the list of original wordle answers is possible.  Once a guess is made, all of the potential feedback values are considered.  Each feedback results in a new state with a subset of the original possible answers.  The game will always select the feedback which keeps this value as large as possible.
"""
  ╠═╡ =#

# ╔═╡ 72ec3834-da5d-4986-84d2-fc5047719aa5
function get_absurdle_feedback(possible_answer_inds::BitVector, guess_index::Integer)
	highest_n = 0
	feedback = 0x00
	@fastmath @inbounds @simd for f in 0x00:0xf2
		n = dot(get_possible_indices(guess_index, f), possible_answer_inds)
		new_high = (n > highest_n)
		feedback = (feedback * !new_high) + (f*new_high)
		highest_n = (highest_n * !new_high) + (n*new_high)
	end
	return feedback, highest_n
end

# ╔═╡ 7f60a260-b073-414a-8f4a-ead768ca2b16
const absurdle_feedback_lookup = Dict{Tuple{WordleState, Int64}, Tuple{UInt8, Int64}}()

# ╔═╡ 711bc605-7b7e-4fc8-9eec-c5afada0c0cd
const absurdle_test_possible_indices = copy(nyt_valid_inds)

# ╔═╡ fa50ef5e-6f9f-42c6-9d04-fc7512c59bd9
function get_absurdle_feedback(s::WordleState, guess_index::Integer; usecache = true, possible_indices = absurdle_test_possible_indices)
	usecache && haskey(absurdle_feedback_lookup, (s, Int64(guess_index))) && return absurdle_feedback_lookup[(s, Int64(guess_index))]
	get_possible_indices!(possible_indices, s; baseline = wordle_original_inds)
	output = get_absurdle_feedback(possible_indices, guess_index)
	if usecache
		absurdle_feedback_lookup[(s, Int64(guess_index))] = output
	end
	return output
end

# ╔═╡ 365f112b-8754-4cb3-8521-51d87f8820bd
function absurdle_prior!(information_gain::Vector{Float32}, s::WordleState; possible_indices = absurdle_test_possible_indices)
	get_possible_indices!(possible_indices, s; baseline = wordle_original_inds)
	base_n = sum(possible_indices)
	starting_entropy = log2(base_n)
	winning_guess = 0
	best_guess = 1
	lowest_entropy = typemax(Float32)
	for i in eachindex(information_gain)
		(feedback, highest_n) = if haskey(absurdle_feedback_lookup, (s, i))
			absurdle_feedback_lookup[(s, i)]
		else
			get_absurdle_feedback(possible_indices, i)
		end
		if feedback == 0xf2
			information_gain .= 0f0
			information_gain[i] = 1f0
			best_guess = i
			break
		end
		final_entropy = log2(highest_n)
		if final_entropy < lowest_entropy
			best_guess = i
			lowest_entropy = final_entropy
		end
		information_gain[i] = Float32(starting_entropy - final_entropy)
	end
	return best_guess
end

# ╔═╡ 2d8ad341-f825-42a9-b796-07a0184f8a89
function absurdle_transition(s::WordleState{N}, guess_index::Int64; kwargs...) where N
	(feedback, n) = get_absurdle_feedback(s, guess_index; kwargs...)
	s′ = WordleState([s.guess_list; UInt16(guess_index)], [s.feedback_list; feedback])
	win = feedback == 0xf2
	score = -1f0
	probabilities = [1f0]
	states = [s′]
	rewards = [score]
	return (rewards, states, probabilities)
end

# ╔═╡ b4b39f69-7c0d-438e-98eb-c07863c8b7e8
const absurdle_transition_distribution = StateMDPTransitionDistribution(absurdle_transition, WordleState())

# ╔═╡ 122e878c-75ad-40ef-9061-004acb7a301a
begin
	absurdle_isterm(s::WordleState{0}) = false
	absurdle_isterm(s::WordleState{N}) where N = s.feedback_list[N] == 0xf2
end

# ╔═╡ b7c1aee0-bff2-4440-a386-41f83809ee9f
const absurdle_mdp = StateMDP(wordle_actions, absurdle_transition_distribution, () -> WordleState(), absurdle_isterm)

# ╔═╡ 0a05e6da-1965-4803-b5f9-1816076b7ee3
function rank_absurdle_guesses(s::WordleState{N}; save_all_scores = false, kwargs...) where N
	if save_all_scores
		l = length(wordle_actions)
		scores = Vector{Float32}(undef, l)
		feedbacks = Vector{UInt8}(undef, l)
		counts = Vector{Int64}(undef, l)
	end
	best_guess_index = first(wordle_actions)
	best_score = 0f0
	@inbounds @simd for guess_index in wordle_actions
		(feedback, n) = get_absurdle_feedback(s, guess_index; kwargs...)
		win = (feedback == 0xf2)
		entropy = log2(n)
		information_gain = wordle_original_entropy - entropy
		score = information_gain / (N + 2 - win) #always rank winning guesses higher than others even if final entropy is 0
		newbest = (score > best_score)
		best_score = (best_score * !newbest) + (score*newbest)
		best_guess_index = (best_guess_index * !newbest) + (guess_index*newbest)

		if save_all_scores
			scores[guess_index] = score
			feedbacks[guess_index] = feedback
			counts[guess_index] = n
		end
	end

	simple_output = (best_guess = best_guess_index, best_score = best_score)
	!save_all_scores && return simple_output
	return (;simple_output..., best_score = best_score, guess_scores = scores, guess_feedback = feedbacks, guess_remaining_counts = counts)
end

# ╔═╡ 5503dee5-32c5-4c11-b761-921fab0fb1bb
function rank_absurdle_guesses!(scores::Vector{Float32}, s::WordleState{N}; kwargs...) where N
	best_guess_index = first(wordle_actions)
	best_score = 0f0
	@inbounds @simd for guess_index in wordle_actions
		(feedback, n) = get_absurdle_feedback(s, guess_index; kwargs...)
		win = (feedback == 0xf2)
		entropy = log2(n)
		information_gain = wordle_original_entropy - entropy
		score = information_gain / (N + 1 - win) #always rank winning guesses higher than others even if final entropy is 0
		newbest = (score > best_score)
		best_score = (best_score * !newbest) + (score*newbest)
		best_guess_index = (best_guess_index * !newbest) + (guess_index*newbest)
		scores[guess_index] = score
	end

	(best_guess = best_guess_index, best_score = best_score, guess_scores = scores)
end

# ╔═╡ 54e18f37-7dfc-4c81-9893-344331b32dd7
const absurdle_scores = zeros(Float32, length(nyt_valid_inds))

# ╔═╡ 9153e796-3185-48fc-bff6-f07078bdfc14
function absurdle_greedy_policy(s::WordleState; absurdle_scores = absurdle_scores, kwargs...)
	# output = rank_absurdle_guesses(s; kwargs...)
	# return Int64(output.best_guess)
	absurdle_prior!(absurdle_scores, s; kwargs...)
end

# ╔═╡ 3153a447-9f56-42a2-aa19-95e8a230ac22
#=╠═╡
md"""
### Absurdle MCTS
"""
  ╠═╡ =#

# ╔═╡ 9101a345-650e-42c6-93d0-515f92f5c72f
function absurdle_greedy_prior!(prior::Vector{Float32}, s::WordleState; kwargs...)
	output = rank_absurdle_guesses!(prior, s; kwargs...)
	return Int64(output.best_guess)
end

# ╔═╡ eb2afee9-6c33-4e66-99eb-0899dedaa2a6
function absurdle_uniform_prior!(prior::Vector{Float32}, s::WordleState; possible_indices = absurdle_test_possible_indices)
	prior .= 1f0
	get_possible_indices!(possible_indices, s; baseline = wordle_original_inds)
	return findfirst(possible_indices)
end

# ╔═╡ ff70b9ad-18e5-4b9f-ade4-888de0fbbc5d
function run_absurdle_mcts(s::WordleState, nsims::Integer; π_dist! = absurdle_prior!, topn = 10, p_scale = 100f0, mdp = absurdle_mdp, kwargs...)
	(mcts_guess, visit_counts, values) = monte_carlo_tree_search(mdp, 1f0, s, π_dist!, p_scale, topn; nsims = nsims, sim_message=true, kwargs...)
	return (visit_counts, values)
end

# ╔═╡ 66bb9ca5-93b1-4bec-8eb7-5581c44b56cd
const (absurdle_root_visits, absurdle_root_values) = run_absurdle_mcts(WordleState(), 1; π_dist! = absurdle_greedy_prior!)

# ╔═╡ 526d423e-c65a-49fd-b15b-e4598254af93
function show_absurdle_mcts_guesses(visit_counts::Dict, values::Dict, s::WordleState)
	inds = visit_counts[s].nzind
	output = rank_absurdle_guesses(s; save_all_scores=true)
	ranked_guess_inds = sortperm(output.guess_scores; rev=true)
	policy_rank_lookup = Dict(zip(ranked_guess_inds, eachindex(ranked_guess_inds)))
	ranked_guesses = sort([(word = nyt_valid_words[i], expected_turns =  -values[s][i], visits = visit_counts[s][i], expected_value = values[s][i], policy_rank = policy_rank_lookup[i]) for i in inds]; by = t -> t.expected_value, rev=true)
	DataFrame(ranked_guesses)
end

# ╔═╡ aaf516b5-f982-44c3-bcae-14d46ad72e82
#=╠═╡
md"""
# Dependencies
"""
  ╠═╡ =#

# ╔═╡ 1faaa157-3d5f-450d-99e7-dae5d70501f9
begin
	 html"""
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

# ╔═╡ ee646ead-3600-49ed-b7ed-3c9a8af7d195
#=╠═╡
md"""
# Wordle Visualization
"""
  ╠═╡ =#

# ╔═╡ 58316d89-7149-42fe-802f-ecd44999ea77
#=╠═╡
md"""
## HTML Elements
"""
  ╠═╡ =#

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
# ╠═╡ skip_as_script = true
#=╠═╡
HTML("""
<div style = "display: grid; align-content: center; justify-items: center; grid-template-columns: repeat(10, 1fr); grid-template-rows: repeat(10, 1fr);">
$(mapreduce(f -> show_pattern(f; sizepct = 0.1), add_elements, feedback_matrix[1:10, 1:10]))
</div>
""")
  ╠═╡ =#

# ╔═╡ 64014b10-9974-475a-8625-fef445fe8e4f
#=╠═╡
md"""
## Single Guess Input Input Element
"""
  ╠═╡ =#

# ╔═╡ e426b806-4435-49b9-824a-04fe34a48e9e
#=╠═╡
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
  ╠═╡ =#

# ╔═╡ 10eb71ed-df57-49b2-ae98-336191346b29
#=╠═╡
md"""
## Playable Game Element
"""
  ╠═╡ =#

# ╔═╡ 014696a0-0568-4944-ad2f-1811b726b2ee
sample_answer() = rand(wordle_original_answers)

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
#=╠═╡
md"""
## Game Input Element
"""
  ╠═╡ =#

# ╔═╡ 1baafa2c-b087-41be-9e5f-6adea8a985fc
#=╠═╡
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
  ╠═╡ =#

# ╔═╡ b9beca61-7efc-4186-81c2-1ceda103c801
#=╠═╡
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
  ╠═╡ =#

# ╔═╡ e460fde6-f65a-4af2-9517-327a187b112a
#=╠═╡
@bind rawanswer confirm(WordleInput(default="apple"))
  ╠═╡ =#

# ╔═╡ 8a090c51-aa61-4090-96a1-8f4833bb9983
#=╠═╡
WordleGame(rawanswer.word; nguesses = 1)
  ╠═╡ =#

# ╔═╡ ee740b4f-93e0-4197-986f-d1ba47d23266
#=╠═╡
@bind greedy_policy_state WordleGameInput()
  ╠═╡ =#

# ╔═╡ 2cda2a2c-caf2-474c-90a4-388f15260501
#=╠═╡
eval_guess_information_gain(WordleState(greedy_policy_state.guesses, greedy_policy_state.feedback); save_all_scores = true) |> display_one_step_guesses |> DataFrame
  ╠═╡ =#

# ╔═╡ 70dcac79-2839-4c1d-8f48-cbe294b3ffb0
#=╠═╡
@bind example_state WordleGameInput()
  ╠═╡ =#

# ╔═╡ bf1c13cd-a066-4865-b43f-0cba3d3df8d2
#=╠═╡
md"""Proposed Guess: $(@bind example_new_guess Select(nyt_valid_words))"""
  ╠═╡ =#

# ╔═╡ 69aacb14-52b2-49a2-a2bc-e8c07b9e1986
#=╠═╡
md"""Adjust Transition State Size: $(@bind size_adj confirm(Slider(100:500, default = 100)))"""
  ╠═╡ =#

# ╔═╡ 83302fc6-49ff-4d25-9fd7-0b491b79fc73
#=╠═╡
md"""
Number of Guesses to Evaluate $(@bind num_guess_evals NumberField(1:length(nyt_valid_inds), default = 10))
"""
  ╠═╡ =#

# ╔═╡ 01be89b2-6233-4e11-b142-fa6ad05880a3
#=╠═╡
begin
	num_guess_evals
	@bind run_policy_iteration CounterButton("Run policy iteration")
end
  ╠═╡ =#

# ╔═╡ 57281f5c-4c93-4308-b1e7-e5a890655262
# ╠═╡ show_logs = false
#=╠═╡
if run_policy_iteration > 0
	evaluate_wordle_state(WordleState(); num_evaluations = num_guess_evals)
else
	md"""Waiting to run policy evaluation for the top $num_guess_evals guesses"""
end
  ╠═╡ =#

# ╔═╡ 5a0ab378-2c88-4851-99a0-ab816457cc6b
#=╠═╡
@bind new_test_state_raw PlutoUI.combine() do Child
	md"""
	Number of Guesses to Evaluate: $(Child(:num_evals, NumberField(1:10000, default = 10)))
	
	$(Child(:state, WordleGameInput()))
	"""
end |> confirm
  ╠═╡ =#

# ╔═╡ 24478951-775b-4e05-88b5-dca8d70e1103
#=╠═╡
const new_test_state = WordleState(new_test_state_raw.state.guesses, new_test_state_raw.state.feedback)
  ╠═╡ =#

# ╔═╡ 9f9b8db2-7c74-49f0-b067-bb6aa433fbe0
#=╠═╡
md"""Show Only Hard Mode Guesses: $(@bind filter_hard CheckBox())"""
  ╠═╡ =#

# ╔═╡ faade165-8066-4deb-93a4-2b7daabd57dc
#=╠═╡
if run_policy_iteration > 0
	evaluate_wordle_state(new_test_state; num_evaluations = new_test_state_raw.num_evals, filter_hard = filter_hard)
else
	md"""Waiting to run policy iteration on $new_test_state"""
end
  ╠═╡ =#

# ╔═╡ 67d76524-e5b6-48bc-9add-2aefd333876d
#=╠═╡
@bind run_root_mcts CounterButton("Run Wordle MCTS from Root State")
  ╠═╡ =#

# ╔═╡ 2bae314a-e97c-41ef-b1a4-187617d9c88b
#=╠═╡
@bind root_mcts_params PlutoUI.combine() do Child
	md"""
	Top N for Prior Sampling: $(Child(:topn, NumberField(1:1000, default = 10)))
	Exploration Constant: $(Child(:c, NumberField(1f0:0.001f0:100f0, default = 1f0)))
	Number of Simulations: $(Child(:nsims, NumberField(1:100_000, default = 100)))
	"""
end |> confirm
  ╠═╡ =#

# ╔═╡ a6c2eb9a-32cb-4343-8e40-fa2b27d4d792
#=╠═╡
if run_root_mcts > 0
	run_wordle_mcts(WordleState(), root_mcts_params.nsims; 
		topn = root_mcts_params.topn, 
		p_scale = 100f0, 
		sim_message = true,
		c = root_mcts_params.c, 
		visit_counts = root_wordle_visit_counts,
		Q = root_wordle_values,
		make_step_kwargs = k -> (possible_indices = test_possible_indices,))
	show_wordle_mcts_guesses(root_wordle_visit_counts, root_wordle_values, WordleState())
else
	md"""
	Showing preliminary results for 1 run.  Waiting to run MCTS for $(root_mcts_params.nsims) simulations
	
	$(show_wordle_mcts_guesses(root_wordle_visit_counts, root_wordle_values, WordleState()))
	"""
end
  ╠═╡ =#

# ╔═╡ 57d62f1e-6213-4328-94ac-b4b3126ddd5b
#=╠═╡
begin
	root_mcts_params
	run_root_mcts
	compare_wordle_polices_over_answers(WordleState(), wordle_greedy_information_gain_π, wordle_root_tree_policy) |> display_policy_compare |> df -> sort(df, "policy2_improvement")
end
  ╠═╡ =#

# ╔═╡ 28b733b6-3989-493a-a00d-24c48da6b338
#=╠═╡
begin
	root_mcts_params
	run_root_mcts
	compare_wordle_polices_over_answers(new_test_state, wordle_greedy_information_gain_π, wordle_root_tree_policy) |> display_policy_compare |> df -> sort(df, "policy2_improvement")
end
  ╠═╡ =#

# ╔═╡ 1c43214e-951d-4b66-8db5-4a90d40ab533
#=╠═╡
begin 
	root_mcts_params
	run_root_mcts
	visited_guesses = root_wordle_visit_counts[WordleState()].nzind
	ranked_visited_inds = sortperm(root_wordle_values[WordleState()][visited_guesses]; rev=true)
	md"""Select a root guess to explore: $(@bind root_guess Select([nyt_valid_words[i] for i in visited_guesses[ranked_visited_inds]]))"""
end
  ╠═╡ =#

# ╔═╡ 7de01236-ba21-4329-815e-beb4633882db
#=╠═╡
const root_transitions = wordle_transition(WordleState(), root_guess)
  ╠═╡ =#

# ╔═╡ c6fdb911-5042-4f88-8484-2dbcd553892e
#=╠═╡
const ranked_transition_inds = sortperm(root_transitions.probabilities; rev=true)
  ╠═╡ =#

# ╔═╡ 3685266e-6ba9-400b-a590-69f930650124
#=╠═╡
const ranked_transition_states = Dict(zip(root_transitions.transition_states[ranked_transition_inds], eachindex(root_transitions.probabilities)))
  ╠═╡ =#

# ╔═╡ e610c03c-4bd9-40e1-a019-87c4fce39602
#=╠═╡
md"""Select transition state: $(@bind explore_state Select([a[2] => string(a[1]) for a in zip(eachindex(ranked_transition_inds), root_transitions.transition_states[ranked_transition_inds])]))"""
  ╠═╡ =#

# ╔═╡ 11fba9cf-1094-4deb-8e1a-ddd781314aa9
#=╠═╡
compare_wordle_polices_over_answers(explore_state, wordle_greedy_information_gain_π, wordle_root_tree_policy) |> display_policy_compare |> df -> sort(df, "policy2_improvement")
  ╠═╡ =#

# ╔═╡ caaeeaee-94b7-498d-a746-a2c5c7177347
#=╠═╡
@bind mcts_reset Button("Click to reset MCTS Evaluation")
  ╠═╡ =#

# ╔═╡ fb8439a3-ee95-487f-a755-ffcbd8b3c381
#=╠═╡
begin 
	mcts_reset
	@bind stop_mcts_eval CounterButton("Click to stop MCTS Wordle Evaluation")
end
  ╠═╡ =#

# ╔═╡ d9d54e56-a32f-4ffb-820d-df0b7918c78c
#=╠═╡
begin
	mcts_reset
	@bind mcts_counter CounterButton("Click to run MCTS Wordle Evaluation")
end
  ╠═╡ =#

# ╔═╡ 7cd9cb06-4731-4eaa-b745-32118278d360
#=╠═╡
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
  ╠═╡ =#

# ╔═╡ 9b1c9eb1-f97d-44f5-9050-9b18ea8814b0
#=╠═╡
@bind run_root_guess_candidate_mcts CounterButton("Run Wordle MCTS from Root State Only Using Candidate Guesses")
  ╠═╡ =#

# ╔═╡ a0ecb0ec-f442-45ca-bc15-9967e82a9905
#=╠═╡
@bind root_guess_candidate_mcts_params PlutoUI.combine() do Child
	md"""
	Top N for Prior Sampling: $(Child(:topn, NumberField(1:1000, default = 10)))
	Number of Simulations: $(Child(:nsims, NumberField(1:100_000, default = 100)))
	"""
end |> confirm
  ╠═╡ =#

# ╔═╡ 12126ca1-b728-4a91-bc53-f0dacd412265
#=╠═╡
@bind mcts_root_candidate_reset Button("Click to reset MCTS Evaluation")
  ╠═╡ =#

# ╔═╡ bfb77472-6bcc-4a9d-9cd1-a7c2686da539
#=╠═╡
begin 
	mcts_root_candidate_reset
	@bind stop_root_candidate_mcts_eval CounterButton("Click to stop MCTS Wordle Evaluation")
end
  ╠═╡ =#

# ╔═╡ d2cf651d-3547-4b80-b837-3b5f297fffa5
#=╠═╡
begin
	mcts_root_candidate_reset
	@bind root_candidate_mcts_counter CounterButton("Click to run MCTS Wordle Evaluation")
end
  ╠═╡ =#

# ╔═╡ 6d79c363-2ff5-4489-9870-3a3813f592f3
#=╠═╡
if stop_root_candidate_mcts_eval > 0
	md"""
	Evaluation stopped.  Wait for loop to end below and then press reset before running
	"""
elseif root_candidate_mcts_counter > 0
	md"""
	Evaluation started
	"""
else
	md"""
	Waiting to run mcts evaluation
	"""
end
  ╠═╡ =#

# ╔═╡ a5befb7c-a1e1-4d55-9b8f-c599748f6f00
#=╠═╡
if root_candidate_mcts_counter > 0
	@use_effect([]) do
		t = time()
		schedule(Task() do
			nruns = 1_000
			nsims = 1_000
			for i in 1:nruns
				stop_root_candidate_mcts_eval > 0 && break	
				elapsed_minutes = (time() - t)/60
				etr = (elapsed_minutes * nruns / i) - elapsed_minutes
				set_root_candidate_run("Running $i of $nruns after $(round(Int64, (time() - t)/60)) minutes.  Estimated $(round(Int64, etr)) minutes left")
				output = @spawn show_afterstate_mcts_guesses(run_wordle_root_candidate_mcts(WordleState(), nsims, root_guess_bit_filter; tree_values = root_guess_candidate_tree, sim_message = false, p_scale = 100f0), WordleState())
				set_root_candidate_mcts_options(fetch(output))
				sleep(.01)
			end
			if stop_root_candidate_mcts_eval > 0
				set_root_candidate_run("Interrupted")
			else
				set_root_candidate_run("Completed after $(round(Int64, (time() - t) / 60)) minutes")
			end
		end)
	end
end
  ╠═╡ =#

# ╔═╡ bd37635d-5f92-4fe4-9245-44a4019fcd54
#=╠═╡
begin
	new_test_state
	@bind run_new_state_mcts CounterButton("Evaluate new state")
end
  ╠═╡ =#

# ╔═╡ 9f132408-31f3-4ad3-a486-f6a94b276f32
#=╠═╡
begin
	new_test_state
	md"""Request additional simulations: $(@bind new_state_num_sims confirm(NumberField(0:10_000, default = 0)))"""
end
  ╠═╡ =#

# ╔═╡ 0a4ac51e-ab8f-467f-9281-691023af400a
#=╠═╡
begin
	visited_new_guesses = root_wordle_visit_counts[new_test_state].nzind
	ranked_new_visited_inds = sortperm(root_wordle_values[new_test_state][visited_new_guesses]; rev=true)
	md"""Select a guess to explore: $(@bind candidate_guess Select([nyt_valid_words[i] for i in visited_new_guesses[ranked_new_visited_inds]]))"""
end
  ╠═╡ =#

# ╔═╡ f82f878f-0dfb-4d87-94ff-1b5564e30f5c
#=╠═╡
@bind run_hardmode_root_mcts CounterButton("Run Hardmode Wordle MCTS from Root State")
  ╠═╡ =#

# ╔═╡ 50a41592-576b-424c-9513-ff1ea4c8ca0f
#=╠═╡
@bind root_hardmode_mcts_params PlutoUI.combine() do Child
	md"""
	Top N for Prior Sampling: $(Child(:topn, NumberField(1:1000, default = 10)))
	Number of Simulations: $(Child(:nsims, NumberField(1:100_000, default = 100)))
	"""
end |> confirm
  ╠═╡ =#

# ╔═╡ 5724fed9-8f1e-4fad-b631-88fe86354e14
#=╠═╡
if run_hardmode_root_mcts > 0
	run_wordle_mcts(WordleState(), root_hardmode_mcts_params.nsims; 
		topn = 10, 
		p_scale = 100f0, 
		sim_message = true,
		c = 1f0, 
		visit_counts = root_wordle_hardmode_visit_counts,
		Q = root_wordle_hardmode_values,
		make_step_kwargs = k -> (possible_indices = test_possible_indices,),
		π_dist! = wordle_hardmode_greedy_information_gain_prior!,
		prior_kwargs = make_hardmode_information_gain_kwargs())
	show_wordle_mcts_guesses(root_wordle_hardmode_visit_counts, root_wordle_hardmode_values, WordleState(); calc_guess_value = wordle_hardmode_greedy_information_gain_guess_value)
else
	md"""
	Showing preliminary results for 1 run.  Waiting to run MCTS for $(root_hardmode_mcts_params.nsims) simulations
	
	$(show_wordle_mcts_guesses(root_wordle_hardmode_visit_counts, root_wordle_hardmode_values, WordleState(); calc_guess_value = wordle_hardmode_greedy_information_gain_guess_value))
	"""
end
  ╠═╡ =#

# ╔═╡ f1166557-4073-4d82-b4c0-db7189e7b381
#=╠═╡
@bind run_button_dontwordle_mcts CounterButton("Run Dontwordle MCTS")
  ╠═╡ =#

# ╔═╡ 9fef413d-773a-432f-a3e0-780ec1431c1c
#=╠═╡
if run_button_dontwordle_mcts > 0
	run_dontwordle_mcts(DontWordleState(), 2, 1_000; visit_counts = dontwordle_root_visits, Q = dontwordle_root_values, c = 2f0)
	show_dontwordle_mcts_guesses(dontwordle_root_visits, dontwordle_root_values, DontWordleState())
end
  ╠═╡ =#

# ╔═╡ aec20c62-092a-4778-a9ed-4a7aacd24d50
#=╠═╡
@bind new_dontwordle_test_state_raw PlutoUI.combine() do Child
	md"""
	Number of Guesses to Evaluate: $(Child(:num_evals, NumberField(1:10000, default = 10)))
	
	$(Child(:state, WordleGameInput()))
	"""
end |> confirm
  ╠═╡ =#

# ╔═╡ c715ce58-caf3-4fe3-b01f-60cc29dff1af
#=╠═╡
const test_dontwordle_state = DontWordleState(WordleState(new_dontwordle_test_state_raw.state.guesses, new_dontwordle_test_state_raw.state.feedback), new_dontwordle_test_state_raw.state.undos |> isempty ? WordleState() : WordleState([a[1] for a in new_dontwordle_test_state_raw.state.undos], [a[2] for a in new_dontwordle_test_state_raw.state.undos]))
  ╠═╡ =#

# ╔═╡ a03295ca-646b-4548-ac62-c6e7a8b4b54d
#=╠═╡
function show_dontwordle_guess_scores(s::DontWordleState, maxundos::Integer)
	output = eval_guess_dontwordle_score(test_dontwordle_state, maxundos; save_all_scores = true)
	ranked_guess_inds = output.valid_guesses[output.ranked_guess_inds]
	ranked_entropies = output.expected_entropy[output.ranked_guess_inds]
	[(word = nyt_valid_words[ranked_guess_inds[i]], expected_entropy = ranked_entropies[i], expected_words_left = 2^(ranked_entropies[i])) for i in eachindex(ranked_guess_inds)] |> DataFrame
end
  ╠═╡ =#

# ╔═╡ 3e2e41f9-95b5-4eaa-81a6-3a9669a55c1a
#=╠═╡
@bind run_button_oneundo_dontwordle_mcts CounterButton("Run Dontwordle MCTS with One Undos")
  ╠═╡ =#

# ╔═╡ 789f5070-dd89-4884-9d7c-6c08843c5711
#=╠═╡
if run_button_oneundo_dontwordle_mcts > 0
	run_dontwordle_mcts(DontWordleState(), 1, 100_000; visit_counts = dontwordle_oneundo_root_visits, Q = dontwordle_oneundo_root_values, mdp = dontwordle_oneundo_mdp, c = 1f0)
	show_dontwordle_mcts_guesses(dontwordle_oneundo_root_visits, dontwordle_oneundo_root_values, DontWordleState(); maxundos = 1)
else
	show_dontwordle_mcts_guesses(dontwordle_oneundo_root_visits, dontwordle_oneundo_root_values, DontWordleState(); maxundos = 1)
end
  ╠═╡ =#

# ╔═╡ 79a43e2d-e34d-4821-b8e3-bbe4aa33cca0
#=╠═╡
@bind run_button_noundo_dontwordle_mcts CounterButton("Run Dontwordle MCTS with No Undos")
  ╠═╡ =#

# ╔═╡ c52a6849-8201-4153-99c0-c46c43622958
#=╠═╡
if run_button_noundo_dontwordle_mcts > 0
	run_dontwordle_mcts(DontWordleState(), 0, 100_000; visit_counts = dontwordle_noundo_root_visits, Q = dontwordle_noundo_root_values, mdp = dontwordle_noundo_mdp, c = 1f0)
	show_dontwordle_mcts_guesses(dontwordle_noundo_root_visits, dontwordle_noundo_root_values, DontWordleState(); maxundos = 0)
else
	show_dontwordle_mcts_guesses(dontwordle_noundo_root_visits, dontwordle_noundo_root_values, DontWordleState(); maxundos = 0)
end
  ╠═╡ =#

# ╔═╡ 5cb34251-a56e-4002-8e25-35932996502a
#=╠═╡
@bind run_button_absurdle_mcts CounterButton("Run Absurdle MCTS from Root State")
  ╠═╡ =#

# ╔═╡ 3e16c985-37e6-49e1-9561-21c79552425c
#=╠═╡
if run_button_absurdle_mcts > 0
	run_absurdle_mcts(WordleState(), 10; visit_counts = absurdle_root_visits, Q = absurdle_root_values, p_scale = 100f0, topn = 10, c = 1f0, π_dist! = absurdle_greedy_prior!)
	show_absurdle_mcts_guesses(absurdle_root_visits, absurdle_root_values, WordleState())
else
	show_absurdle_mcts_guesses(absurdle_root_visits, absurdle_root_values, WordleState())
end
  ╠═╡ =#

# ╔═╡ f22fb00d-37f0-4a83-bf60-048b936bf04d
#=╠═╡
@bind absurdle_test_state_raw PlutoUI.combine() do Child
	md"""
	$(Child(:state, WordleGameInput()))
	"""
end
  ╠═╡ =#

# ╔═╡ 6c532310-1f80-419a-a74b-7814bad6a354
#=╠═╡
const absurdle_test_state = WordleState(absurdle_test_state_raw.state.guesses, absurdle_test_state_raw.state.feedback)
  ╠═╡ =#

# ╔═╡ a79b430a-84d7-4253-bfc5-ffb200b98767
#=╠═╡
begin
	run_button_absurdle_mcts
	absurdle_answers_left = get_possible_indices(absurdle_test_state; baseline = nyt_valid_inds)
	if haskey(absurdle_root_visits, absurdle_test_state)
		show_absurdle_mcts_guesses(absurdle_root_visits, absurdle_root_values, absurdle_test_state)
	else
		nyt_valid_words[absurdle_answers_left]
	end
end
  ╠═╡ =#

# ╔═╡ 729ef5f5-1538-48dc-b028-2bb112921fb2
#=╠═╡
@bind testgame WordleGame()
  ╠═╡ =#

# ╔═╡ 8d328017-7f7a-45d0-90bc-77ddd036101f
#=╠═╡
testgame
  ╠═╡ =#

# ╔═╡ d7b0a4ba-ba18-41ad-ade3-dde119f08a13
#=╠═╡
md"""
## Display Games
"""
  ╠═╡ =#

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
#=╠═╡
show_wordle_game([example_guess], [example_guess_feedback_bytes]; truncate = true)
  ╠═╡ =#

# ╔═╡ d2bf55fa-ac04-45d6-9d6a-8c7ac855c3a1
function visualize_wordle_transition(s::WordleState, guess; sizepct = 100)
	(rewards, new_states, probabilities) = wordle_transition(s, guess)
	HTML("""
	<div style = "display: flex; height: 600px;">
	<div>
	<div style = "font-size: 1.5em;"><b>Root State</b></div>
	<div>$(show_wordle_game(s).content)</div>
	</div>
	<div>
	<div style = "font-size: 1.5em;"><b>$(length(new_states)) Transition States (size proportional to probability)</b></div>
	<div style = "display:flex; flex-wrap: wrap; ">
		$(mapreduce(a -> show_wordle_game(WordleState(a[1].guess_list[end:end], a[1].feedback_list[end:end]); truncate = true, sizepct = round(Int64, a[2]*sizepct)).content, add_elements, zip(new_states, probabilities)))
	</div>
	</div>
	</div>
	""")
end

# ╔═╡ 52faa94f-e7ca-4e28-9ed2-0a788db5e231
#=╠═╡
visualize_wordle_transition(isempty(example_state.guesses) ? WordleState() : WordleState(example_state.guesses, convert_bytes.(example_state.feedback)), example_new_guess; sizepct = size_adj)
  ╠═╡ =#

# ╔═╡ 6957a643-0240-425b-b167-ae290b696f16
Base.show(s::WordleState; kwargs...) = show_wordle_game(s; sizepct = 20, truncate = true, kwargs...)

# ╔═╡ 052913b1-1f6f-49b6-bf4d-cefef430fea9
Base.display(s::WordleState; kwargs...) = show_wordle_game(s; sizepct = 20, truncate = true, kwargs...)

# ╔═╡ 2ae69afa-42e5-4c3e-ada0-a71cb288dfb0
#=╠═╡
function display_word_groups(s::WordleState, π::Function)
	word_group_analysis = analyze_wordle_policy_over_answers(s, π)
	@htl("""
	<div style = "font: 2em bold;">Starting State</div>
	$(display(s))
	<hr>
	<div style = "display: flex;">
	<div style = "width: 7em;">Number of Turns</div>
	<div style = "width: 8em;">Remaining Answers</div>
	<div style = "width: 50em;">Answer Words</div>
	</div>
	<hr>
	$(HTML(mapreduce(i -> display_word_group(i, word_group_analysis[i]), add_elements, 1:6)))
	""")
end
  ╠═╡ =#

# ╔═╡ 6abd318a-9c0b-457e-aac2-c70a580c66cd
#=╠═╡
display_word_groups(explore_state, wordle_greedy_information_gain_π)
  ╠═╡ =#

# ╔═╡ e08de788-8947-4882-ae51-f7cbb0daa83b
#=╠═╡
display_word_groups(explore_state, wordle_root_tree_policy)
  ╠═╡ =#

# ╔═╡ 2c1a68f8-19e7-4224-8c26-0f5704b07389
#=╠═╡
@htl("""
<div style = "height: 600px;">
$(HTML(mapreduce(add_elements, setdiff(1:length(ranked_transition_inds), [ranked_transition_states[k] for k in filter(isterm, keys(ranked_transition_states))])) do rank
	i = ranked_transition_inds[rank]
	p = root_transitions.probabilities[i]
	s = root_transitions.transition_states[i]
	possible_indices = get_possible_indices(s)
	l = sum(possible_indices)
	policy_value = wordle_greedy_information_gain_state_value(s; possible_indices = possible_indices)
	tree_value = maximum(root_wordle_values[s][i] for i in root_wordle_visit_counts[s].nzind)
	maximum_value = maximum_possible_score(s; possible_indices = possible_indices)
	possible_improvement = maximum_value - policy_value
	if possible_improvement < 1f-5
		possible_improvement = 0f0
	end
	tree_improvement = tree_value - policy_value
	"""
	<div style = "display: flex;">
	<div style = "width: 4em;">$rank</div>
	<div style = "width: 10em;">$(display(s).content)</div>
	<div style = "width: 6em;">$l</div>
	<div style = "width: 6em;">$(round(Int64, sum(root_wordle_visit_counts[s]) / l))</div>
	<div style = "width: 6em;">$(round(-tree_value; sigdigits = 3))</div>
	<div style = "width: 6em;">$(round(-policy_value; sigdigits = 3))</div>
	<div style = "width: 6em;">$(round(-maximum_value; sigdigits = 3))</div>
	<div style = "width: 7em;">$(round(possible_improvement; sigdigits = 3))</div>
	<div style = "width: 7em;">$(round(tree_improvement; sigdigits = 3))</div>
	<div style = "width: $(round(1000*p))px; background-color: blue; border: 1px solid black;">$(round(p*100; sigdigits = 2))</div>
	</div>
	"""
end))
</div>
""")
  ╠═╡ =#

# ╔═╡ a00b4f9b-eb74-4fa1-bc70-a97844b59022
#=╠═╡
function display_transition_states(s::WordleState, guess_candidate::Integer)
	transitions = wordle_transition(s, guess_candidate)
	ranked_transition_inds = sortperm(transitions.probabilities; rev=true)
	ranked_transition_states = Dict(zip(transitions.transition_states[ranked_transition_inds], eachindex(transitions.probabilities)))
	@htl("""
	<div style = "height: 600px;">
	$(HTML(mapreduce(add_elements, setdiff(1:length(ranked_transition_inds), [ranked_transition_states[k] for k in filter(isterm, keys(ranked_transition_states))])) do rank
		i = ranked_transition_inds[rank]
		p = transitions.probabilities[i]
		s = transitions.transition_states[i]
		possible_indices = get_possible_indices(s)
		l = sum(possible_indices)
		policy_value = wordle_greedy_information_gain_state_value(s; possible_indices = possible_indices)
		tree_value = maximum(root_wordle_values[s][i] for i in root_wordle_visit_counts[s].nzind)
		maximum_value = maximum_possible_score(s; possible_indices = possible_indices)
		possible_improvement = maximum_value - policy_value
		if possible_improvement < 1f-5
			possible_improvement = 0f0
		end
		tree_improvement = tree_value - policy_value
		"""
		<div style = "display: flex;">
		<div style = "width: 4em;">$rank</div>
		<div style = "width: 10em;">$(display(s).content)</div>
		<div style = "width: 6em;">$l</div>
		<div style = "width: 6em;">$(round(Int64, sum(root_wordle_visit_counts[s]) / l))</div>
		<div style = "width: 6em;">$(round(-tree_value; sigdigits = 3))</div>
		<div style = "width: 6em;">$(round(-policy_value; sigdigits = 3))</div>
		<div style = "width: 6em;">$(round(-maximum_value; sigdigits = 3))</div>
		<div style = "width: 7em;">$(round(possible_improvement; sigdigits = 3))</div>
		<div style = "width: 7em;">$(round(tree_improvement; sigdigits = 3))</div>
		<div style = "width: $(round(1000*p))px; background-color: blue; border: 1px solid black;">$(round(p*100; sigdigits = 2))</div>
		</div>
		"""
	end))
	</div>
	""")
end
  ╠═╡ =#

# ╔═╡ 90c5ee33-b6ae-4cde-b554-cedba77c0e0c
#=╠═╡
display_transition_states(new_test_state, word_index[candidate_guess])
  ╠═╡ =#

# ╔═╡ 4d3a8b0a-4116-4cc6-9ef4-bc1a405db8bb
#=╠═╡
function explore_tree_state(s::WordleState, full_tree_visits::Dict, full_tree_values::Dict)
	transition_probability = root_transitions.probabilities[findfirst(s′ == s for s′ in root_transitions.transition_states)]
	most_probable_ind = argmax(root_transitions.probabilities)
	most_probable_state = root_transitions.transition_states[most_probable_ind]
	probability_ranking = ranked_transition_states[s]
	possible_words = get_possible_words(s)
	# tree_visits = full_tree_visits[s]
	# inds = tree_visits.nzind
	# tree_values = full_tree_values[s][inds]
	# ranked_inds = sortperm(tree_values; rev = true)
	# ranked_guesses = [nyt_valid_words[inds[i]] for i in ranked_inds]
	# ranked_values = tree_values[ranked_inds]
	greedy_information_gain_guess = wordle_greedy_information_gain_π(s)
	greedy_guess = nyt_valid_words[greedy_information_gain_guess]
	greedy_value = full_tree_values[s][greedy_information_gain_guess]
	# output = DataFrame((Explored_Guesses = ranked_guesses[i], Tree_Values = ranked_values[i], Tree_Visits = tree_visits.nzind[ranked_inds[i]], Greedy_Guess = ranked_guesses[i] == greedy_guess) for i in eachindex(ranked_values))
	md"""
	Most Probable Feedback State with Probability $(root_transitions.probabilities[most_probable_ind]):

	$(display(most_probable_state))

	Selected State: 
	
	$(display(s)) with probability $transition_probability and ranking $probability_ranking

	 $(length(possible_words)) Answers Left:

	$possible_words

	Explored Guesses:

	Greedy guess is $greedy_guess with a value of $greedy_value
	
	$(show_wordle_mcts_guesses(full_tree_visits, full_tree_values, s))
	"""
end
  ╠═╡ =#

# ╔═╡ e1782372-cb2b-440c-80f1-27d42f31bf57
#=╠═╡
if run_new_state_mcts > 0
	if new_state_num_sims > 0
		(new_visit_counts, new_q) = run_wordle_mcts(new_test_state, new_state_num_sims; 
			topn = 10, 
			p_scale = 100f0, 
			sim_message = true,
			c = 10f0, 
			make_step_kwargs = k -> (possible_indices = test_possible_indices,))
		show_wordle_mcts_guesses(new_visit_counts, new_q, new_test_state)
	else
		show_wordle_mcts_guesses(root_wordle_visit_counts, root_wordle_values, new_test_state)
	end
else
	md"""
	Waiting to evaluate state $(display(new_test_state))
	"""
end
  ╠═╡ =#

# ╔═╡ 0d712e31-4554-4f12-bd02-4386b5d06607
#=╠═╡
function show_dontwordle_game(s::DontWordleState)
	guesses_left = get_possible_indices(s.guesses; baseline = dontwordle_valid_inds)
	answers_left = get_possible_indices(s.undos; baseline = guesses_left)
	answers_left .*= wordle_original_inds
	words_left = if sum(answers_left) < 10
		mapreduce(a -> nyt_valid_words[a], (a, b) -> "$a $b", findall(answers_left))
	else
		""
	end
	@htl("""
	<div style = "display: flex;">
	<div>Guesses $(show_wordle_game(test_dontwordle_state.guesses; sizepct = 50))</div>
	<div>Undos $(show_wordle_game(test_dontwordle_state.undos; sizepct = 50))</div>
	</div>
	<div>$(sum(guesses_left)) valid guesses remain with $(sum(answers_left)) possible answer(s) $words_left</div>
	""")
end
  ╠═╡ =#

# ╔═╡ 1593e79d-8571-4848-9ed4-9e44416dbd49
#=╠═╡
begin
	new_dontwordle_test_state_raw
	@htl("""
	<div style = "display: flex;">
	<div>$(show_dontwordle_game(test_dontwordle_state))</div>
	
	<div>$(show_dontwordle_mcts_guesses(dontwordle_oneundo_root_visits, dontwordle_oneundo_root_values, test_dontwordle_state; maxundos = 1))</div>
	</div>
	""")
end
  ╠═╡ =#

# ╔═╡ eabcf79e-67d3-478f-abbc-cd8ba04138e5
runepisode(absurdle_mdp; π = absurdle_greedy_policy)[4] |> show_wordle_game

# ╔═╡ c809ed82-919c-4a7e-9acb-664499859760
#=╠═╡
md"""
# Other Word Data
"""
  ╠═╡ =#

# ╔═╡ c9ab058c-a3db-4fcd-8ff9-2029e79fbfc6
lookup_dict(dict, key, default) = haskey(dict, key) ? dict[key] : default

# ╔═╡ ea259831-7c83-4780-8670-ef997b51fea0
const includewords = Set(["laser", "lemma", "pious", "misty", "petit", "anion", "alias", "boron", "aloha", "ilium", "swipe", "squid", "cocky", "liter", "algal", "saber", "tuple", "bogus", "combo", "servo", "halal", "agora", "codon", "tapas", "levee", "largo", "chink", "scoot", "xerox", "exons", "prana", "toner", "clunk", "yucca", "bruin", "exude"])

# ╔═╡ ada75047-17da-4857-9c18-b4d696a50681
const wordcdf = eachindex(wordle_original_answers) ./ length(wordle_original_answers)

# ╔═╡ 4b0fd69e-682b-47f5-852a-7dbe8bc1a1dd
wordle_cdf(rank::Integer) = erf(0.00028355891696184375*rank)

# ╔═╡ d025366f-eb5c-4bb0-b8d7-24132e2c3d27
wordle_pdf(rank::Integer) = 0.00028355891696184375*2/sqrt(pi) * exp(-(0.00028355891696184375*rank)^2)


# ╔═╡ b36e6991-7292-4290-9e28-85430fc0e897
# occurence frequencies taken from https://raw.githubusercontent.com/3b1b/videos/master/_2022/wordle/data/freq_map.json
const word_frequencies = Dict("aahed"=> 4.501494e-08, "aalii"=> 2.955076e-10, "aargh"=> 3.474266e-08, "aarti"=> 1.0270499999999999e-07, "abaca"=> 4.172372000000001e-08, "abaci"=> 2.968276e-08, "aback"=> 2.699342e-06, "abacs"=> 3.1059160000000005e-10, "abaft"=> 1.1280262e-07, "abaka"=> 5.2003219999999995e-09, "abamp"=> 1.4229644e-10, "aband"=> 1.842324e-08, "abase"=> 9.613907999999998e-08, "abash"=> 2.5823639999999998e-08, "abask"=> 2.02239802e-09, "abate"=> 8.468224e-07, "abaya"=> 8.439948e-08, "abbas"=> 1.4686259999999999e-06, "abbed"=> 2.3027160000000005e-09, "abbes"=> 3.522334e-08, "abbey"=> 6.393404000000001e-06, "abbot"=> 3.5446e-06, "abcee"=> 1.1574035999999998e-10, "abeam"=> 9.054506e-08, "abear"=> 1.6375280000000003e-08, "abele"=> 9.327036e-08, "abers"=> 1.9409319999999998e-08, "abets"=> 5.7714539999999997e-08, "abhor"=> 5.68669e-07, "abide"=> 4.437534000000001e-06, "abies"=> 1.652658e-07, "abled"=> 1.530008e-07, "abler"=> 1.114481e-07, "ables"=> 9.486708000000001e-08, "ablet"=> 2.8223260000000005e-09, "ablow"=> 2.325952e-08, "abmho"=> 1.3739468e-10, "abode"=> 3.332456e-06, "abohm"=> 3.478807e-10, "aboil"=> 4.706077999999999e-09, "aboma"=> 1.1429096e-09, "aboon"=> 2.51492e-08, "abord"=> 1.405545e-07, "abore"=> 1.9253066e-09, "abort"=> 7.514977999999999e-07, "about"=> 0.001446756, "above"=> 0.0002028008, "abram"=> 2.4312239999999995e-06, "abray"=> 3.6757620000000004e-09, "abrim"=> 8.363642000000001e-09, "abrin"=> 2.258936e-08, "abris"=> 7.687314e-09, "absey"=> 2.230168e-09, "absit"=> 1.8070659999999998e-08, "abuna"=> 2.523742e-08, "abune"=> 2.1898679999999997e-08, "abuse"=> 3.48896e-05, "abuts"=> 1.264882e-07, "abuzz"=> 1.927152e-07, "abyes"=> 1.208623e-10, "abysm"=> 2.191922e-08, "abyss"=> 3.55745e-06, "acais"=> 2.6784520000000003e-10, "acari"=> 9.260898e-08, "accas"=> 8.249184599999999e-10, "accoy"=> 2.0457443999999998e-10, "acerb"=> 4.64374e-09, "acers"=> 1.3396340000000002e-08, "aceta"=> 2.0212079999999997e-09, "achar"=> 5.8190339999999996e-08, "ached"=> 4.536534e-06, "aches"=> 1.646678e-06, "achoo"=> 3.440584e-08, "acids"=> 1.6505139999999998e-05, "acidy"=> 8.11673e-09, "acing"=> 4.97416e-08, "acini"=> 1.212062e-07, "ackee"=> 3.5917460000000004e-08, "acker"=> 4.4313500000000006e-07, "acmes"=> 4.6539259999999994e-09, "acmic"=> 4.7694318e-10, "acned"=> 6.703428e-09, "acnes"=> 1.0338555999999999e-07, "acock"=> 2.251304e-08, "acold"=> 3.746208e-09, "acorn"=> 8.802106000000002e-07, "acred"=> 1.3091266000000002e-08, "acres"=> 9.162212000000001e-06, "acrid"=> 8.951414000000001e-07, "acros"=> 1.8775839999999997e-08, "acted"=> 1.767344e-05, "actin"=> 1.791298e-06, "acton"=> 8.742459999999998e-07, "actor"=> 1.7857680000000002e-05, "acute"=> 3.285576e-05, "acyls"=> 3.9796299999999994e-09, "adage"=> 9.832024e-07, "adapt"=> 1.099978e-05, "adaws"=> 2.00319886e-09, "adays"=> 1.6723804e-08, "adbot"=> 1.425943e-10, "addax"=> 1.781184e-08, "added"=> 0.0001028232, "adder"=> 7.37006e-07, "addio"=> 5.6324400000000004e-08, "addle"=> 6.202047999999999e-08, "adeem"=> 7.99255e-09, "adept"=> 2.5460960000000004e-06, "adhan"=> 3.062196e-08, "adieu"=> 1.211535e-06, "adios"=> 1.3868040000000002e-07, "adits"=> 2.235742e-08, "adman"=> 3.6445780000000004e-08, "admen"=> 1.5153480000000004e-08, "admin"=> 1.810436e-06, "admit"=> 2.819082e-05, "admix"=> 1.2917690000000002e-08, "adobe"=> 1.8630220000000002e-06, "adobo"=> 1.283168e-07, "adopt"=> 1.801588e-05, "adore"=> 2.045672e-06, "adorn"=> 1.0802664000000001e-06, "adown"=> 1.0014458e-07, "adoze"=> 5.805744e-10, "adrad"=> 2.8924760000000003e-09, "adred"=> 9.197842000000002e-10, "adsum"=> 1.1177524e-08, "aduki"=> 9.556678e-09, "adult"=> 4.754738e-05, "adunc"=> 2.0329593999999997e-10, "adust"=> 1.3730560000000001e-08, "advew"=> 6.686342000000001e-11, "adyta"=> 3.877192e-09, "adzed"=> 5.042798000000001e-09, "adzes"=> 6.191178000000001e-08, "aecia"=> 3.536386e-09, "aedes"=> 4.33398e-07, "aegis"=> 8.62897e-07, "aeons"=> 2.5726199999999996e-07, "aerie"=> 2.016578e-07, "aeros"=> 4.557548e-08, "aesir"=> 8.476254e-08, "afald"=> 2.1472129999999998e-10, "afara"=> 6.025464e-09, "afars"=> 9.937821999999999e-09, "afear"=> 3.092228e-09, "affix"=> 5.534328e-07, "afire"=> 4.173978e-07, "aflaj"=> 1.0743036e-08, "afoot"=> 1.0016201999999999e-06, "afore"=> 1.2924872e-06, "afoul"=> 3.6934540000000005e-07, "afrit"=> 4.90581e-08, "afros"=> 3.219652e-08, "after"=> 0.0008462688000000001, "again"=> 0.00046067580000000007, "agama"=> 9.1871e-08, "agami"=> 1.313132e-08, "agape"=> 1.0922115999999998e-06, "agars"=> 1.955158e-08, "agast"=> 7.761402e-09, "agate"=> 4.590844e-07, "agave"=> 5.82439e-07, "agaze"=> 3.999662e-09, "agene"=> 5.0428399999999995e-09, "agent"=> 5.331242e-05, "agers"=> 1.426656e-07, "agger"=> 9.205498e-08, "aggie"=> 9.793444e-07, "aggri"=> 4.065998e-10, "aggro"=> 5.426242e-08, "aggry"=> 8.98589e-10, "aghas"=> 1.3691519999999999e-08, "agila"=> 9.345652e-09, "agile"=> 3.85706e-06, "aging"=> 1.45212e-05, "agios"=> 1.1374950000000002e-07, "agism"=> 3.0187080000000004e-09, "agist"=> 4.596938e-09, "agita"=> 1.951662e-08, "aglee"=> 1.8224986000000002e-09, "aglet"=> 6.479142e-09, "agley"=> 1.2791440000000001e-08, "agloo"=> 1.69067e-10, "aglow"=> 4.610106e-07, "aglus"=> 1.9935578e-10, "agmas"=> 3.059788e-10, "agoge"=> 1.1102676e-08, "agone"=> 1.0712426000000002e-07, "agons"=> 4.215392e-09, "agony"=> 7.544356e-06, "agood"=> 2.220942e-08, "agora"=> 8.215696e-07, "agree"=> 4.221286e-05, "agria"=> 1.6921062000000003e-08, "agrin"=> 3.1771979999999995e-08, "agros"=> 2.610848e-08, "agued"=> 5.1871e-09, "agues"=> 4.9502139999999994e-08, "aguna"=> 2.7922180000000003e-09, "aguti"=> 3.354304e-09, "ahead"=> 5.2857239999999995e-05, "aheap"=> 3.2794859999999998e-09, "ahent"=> 7.911551999999999e-10, "ahigh"=> 7.700074e-09, "ahind"=> 4.847116e-09, "ahing"=> 4.344278e-09, "ahint"=> 2.2137960000000003e-08, "ahold"=> 5.089674e-07, "ahull"=> 3.1200180000000008e-09, "ahuru"=> 2.0160714e-09, "aidas"=> 3.93298e-09, "aided"=> 5.901974e-06, "aider"=> 1.198656e-07, "aides"=> 1.585398e-06, "aidoi"=> 1.3963712e-10, "aidos"=> 1.6382460000000002e-08, "aiery"=> 1.7864303999999997e-09, "aigas"=> 5.584584e-09, "aight"=> 7.006802e-08, "ailed"=> 1.6689779999999998e-07, "aimed"=> 2.1960580000000003e-05, "aimer"=> 1.2015506e-07, "ainee"=> 1.3178802e-09, "ainga"=> 5.899670000000001e-10, "aioli"=> 1.682648e-07, "aired"=> 1.657228e-06, "airer"=> 7.950056000000001e-09, "airns"=> 2.214134e-09, "airth"=> 3.245508e-08, "airts"=> 5.0394100000000005e-09, "aisle"=> 5.122444e-06, "aitch"=> 3.648552e-08, "aitus"=> 5.614985999999999e-10, "aiver"=> 1.506348e-09, "aiyee"=> 3.3374800000000003e-09, "aizle"=> 5.7634634e-10, "ajies"=> 1.7564038000000001e-10, "ajiva"=> 4.15095e-09, "ajuga"=> 2.060942e-08, "ajwan"=> 7.374024e-10, "akees"=> 2.461278e-10, "akela"=> 1.3980772000000002e-07, "akene"=> 6.11016e-10, "aking"=> 8.756932e-07, "akita"=> 1.56542e-07, "akkas"=> 5.963728000000001e-09, "alaap"=> 4.914198e-09, "alack"=> 1.798444e-07, "alamo"=> 6.082996e-07, "aland"=> 1.868644e-07, "alane"=> 3.4427e-08, "alang"=> 3.2426634e-07, "alans"=> 7.455594e-08, "alant"=> 7.081957999999999e-09, "alapa"=> 2.3648719999999997e-09, "alaps"=> 4.841493999999999e-10, "alarm"=> 1.5956980000000002e-05, "alary"=> 1.8179479999999998e-08, "alate"=> 1.5917526e-08, "alays"=> 1.1231674e-09, "albas"=> 1.2415920000000001e-08, "albee"=> 2.1030839999999997e-07, "album"=> 8.152288e-06, "alcid"=> 3.87807e-09, "alcos"=> 2.202318e-09, "aldea"=> 8.028054e-08, "alder"=> 1.0886682000000001e-06, "aldol"=> 2.54654e-07, "aleck"=> 2.894226e-07, "alecs"=> 7.039036000000001e-09, "alefs"=> 1.2733752000000001e-09, "aleft"=> 3.240508e-09, "aleph"=> 1.5608768e-06, "alert"=> 1.379888e-05, "alews"=> 4.784592e-11, "aleye"=> 1.954518e-10, "alfas"=> 8.330604e-09, "algae"=> 3.508284e-06, "algal"=> 1.2983599999999999e-06, "algas"=> 7.024584000000001e-09, "algid"=> 9.39403e-09, "algin"=> 1.4901862000000002e-08, "algor"=> 2.2976e-08, "algum"=> 8.387819999999999e-08, "alias"=> 1.766534e-06, "alibi"=> 1.534254e-06, "alien"=> 1.2803640000000002e-05, "alifs"=> 1.484522e-09, "align"=> 5.3821379999999995e-06, "alike"=> 1.421084e-05, "aline"=> 4.6352e-07, "alist"=> 4.172223999999999e-08, "alive"=> 4.459193999999999e-05, "aliya"=> 1.6263399999999997e-07, "alkie"=> 8.494644e-09, "alkos"=> 1.1114986000000001e-10, "alkyd"=> 8.458115999999999e-08, "alkyl"=> 1.51327e-06, "allay"=> 7.279446000000001e-07, "allee"=> 1.824824e-07, "allel"=> 1.8513154e-08, "alley"=> 7.379884000000001e-06, "allis"=> 1.8760700000000002e-07, "allod"=> 4.97088e-09, "allot"=> 2.6216939999999997e-07, "allow"=> 8.98091e-05, "alloy"=> 5.504686e-06, "allyl"=> 3.3172780000000003e-07, "almah"=> 2.1239099999999995e-08, "almas"=> 9.799878e-08, "almeh"=> 1.4077576e-09, "almes"=> 3.1141260000000004e-08, "almud"=> 7.466204e-09, "almug"=> 1.9927348000000002e-08, "alods"=> 1.2605460000000001e-09, "aloed"=> 2.5237040000000005e-10, "aloes"=> 2.0408420000000003e-07, "aloft"=> 2.367518e-06, "aloha"=> 3.46051e-07, "aloin"=> 1.0002428e-08, "alone"=> 0.00013902480000000002, "along"=> 0.000221563, "aloof"=> 1.9707719999999995e-06, "aloos"=> 3.3013882000000003e-10, "aloud"=> 1.154556e-05, "alowe"=> 2.3488756e-09, "alpha"=> 1.2388300000000001e-05, "altar"=> 1.323824e-05, "alter"=> 1.361182e-05, "altho"=> 1.58569e-07, "altos"=> 2.1349779999999996e-07, "alula"=> 4.016382e-08, "alums"=> 5.460924e-08, "alure"=> 1.2351162e-09, "alvar"=> 2.6730399999999997e-07, "alway"=> 3.14249e-07, "amahs"=> 1.1710881999999999e-08, "amain"=> 8.098022e-08, "amass"=> 4.915587999999999e-07, "amate"=> 3.114522e-08, "amaut"=> 6.708988e-10, "amaze"=> 7.116605999999999e-07, "amban"=> 2.3386719999999996e-08, "amber"=> 8.013106e-06, "ambit"=> 6.034956e-07, "amble"=> 2.796046e-07, "ambos"=> 2.5564880000000003e-07, "ambry"=> 1.2213958000000003e-08, "ameba"=> 3.456052000000001e-08, "ameer"=> 9.809898e-08, "amend"=> 2.584402e-06, "amene"=> 3.9168564e-08, "amens"=> 5.8515359999999994e-08, "ament"=> 5.141362e-08, "amias"=> 4.457422e-08, "amice"=> 3.43058e-08, "amici"=> 3.02787e-07, "amide"=> 8.794438000000001e-07, "amido"=> 6.69862e-08, "amids"=> 3.4208179999999995e-09, "amies"=> 4.511814e-08, "amiga"=> 1.5832706000000002e-07, "amigo"=> 4.893554000000001e-07, "amine"=> 1.7799659999999998e-06, "amino"=> 1.148976e-05, "amins"=> 5.628536e-09, "amirs"=> 7.367058e-08, "amiss"=> 1.8123099999999998e-06, "amity"=> 7.223352000000001e-07, "amlas"=> 1.4314965999999998e-09, "amman"=> 6.600408e-07, "ammon"=> 1.2958308e-06, "ammos"=> 1.2787318e-08, "amnia"=> 1.7987136000000002e-09, "amnic"=> 1.1179981999999999e-10, "amnio"=> 1.6074419999999997e-08, "amoks"=> 6.030566e-10, "amole"=> 1.0182054e-08, "among"=> 0.00025544060000000005, "amort"=> 9.3274e-09, "amour"=> 1.106672e-06, "amove"=> 4.018648e-09, "amowt"=> 1.4819482e-10, "amped"=> 1.735982e-07, "ample"=> 6.631154000000001e-06, "amply"=> 1.296924e-06, "ampul"=> 3.0194820000000002e-09, "amrit"=> 1.289367e-07, "amuck"=> 1.1005881999999999e-07, "amuse"=> 2.0447e-06, "amyls"=> 1.2495236000000002e-09, "anana"=> 2.0759e-08, "anata"=> 3.01154e-08, "ancho"=> 1.0979459999999999e-07, "ancle"=> 1.8121538e-08, "ancon"=> 2.3355360000000002e-08, "andro"=> 8.994575999999999e-08, "anear"=> 2.228178e-08, "anele"=> 9.213104e-09, "anent"=> 9.619452000000001e-08, "angas"=> 3.3227699999999995e-08, "angel"=> 2.3742260000000002e-05, "anger"=> 4.4622119999999996e-05, "angle"=> 3.16747e-05, "anglo"=> 1.0774454e-05, "angry"=> 4.058566e-05, "angst"=> 1.34869e-06, "anigh"=> 8.760174e-08, "anile"=> 1.1665374e-08, "anils"=> 2.6112411999999997e-09, "anima"=> 1.0154103999999999e-06, "anime"=> 7.225968e-07, "animi"=> 1.271142e-07, "anion"=> 1.897262e-06, "anise"=> 6.132204e-07, "anker"=> 2.07566e-07, "ankhs"=> 6.687468e-09, "ankle"=> 8.598192e-06, "ankus"=> 1.8473014e-08, "anlas"=> 1.1946336e-09, "annal"=> 7.5511e-08, "annas"=> 5.25693e-07, "annat"=> 3.8726179999999997e-08, "annex"=> 4.5056460000000005e-06, "annoy"=> 1.374472e-06, "annul"=> 3.806556e-07, "anoas"=> 3.8044200000000005e-10, "anode"=> 2.5620199999999995e-06, "anole"=> 3.0737339999999995e-08, "anomy"=> 2.0380120000000003e-08, "ansae"=> 2.846496e-09, "antae"=> 1.1340936e-08, "antar"=> 9.301900000000001e-08, "antas"=> 1.665888e-08, "anted"=> 3.098658e-08, "antes"=> 7.26855e-07, "antic"=> 1.675442e-07, "antis"=> 9.8177e-08, "antra"=> 1.7806440000000002e-08, "antre"=> 2.3861425999999998e-08, "antsy"=> 3.535282e-07, "anura"=> 3.667206e-08, "anvil"=> 9.730494e-07, "anyon"=> 8.09228e-08, "aorta"=> 2.842526e-06, "apace"=> 5.40662e-07, "apage"=> 4.3323480000000004e-09, "apaid"=> 3.599154e-09, "apart"=> 5.1347399999999996e-05, "apayd"=> 1.336029e-09, "apays"=> 1.4447964e-10, "apeak"=> 7.585866e-09, "apeek"=> 2.330579e-10, "apers"=> 2.5808240000000002e-08, "apert"=> 5.5342240000000004e-08, "apery"=> 2.687884e-09, "apgar"=> 2.484128e-07, "aphid"=> 4.014522e-07, "aphis"=> 2.89647e-07, "apian"=> 2.391368e-08, "aping"=> 1.433626e-07, "apiol"=> 5.0994300000000005e-09, "apish"=> 3.922780000000001e-08, "apism"=> 6.388207999999999e-10, "apnea"=> 1.669324e-06, "apode"=> 6.680578e-10, "apods"=> 8.101413999999999e-10, "apoop"=> 4.604776e-11, "aport"=> 8.808386e-09, "appal"=> 5.296396e-08, "appay"=> 3.51572e-09, "appel"=> 4.849286e-07, "apple"=> 2.054572e-05, "apply"=> 5.697018e-05, "appro"=> 8.623589999999999e-08, "appui"=> 5.1472359999999995e-08, "appuy"=> 5.973538e-10, "apres"=> 5.936624e-08, "apron"=> 4.063688e-06, "apses"=> 5.5935179999999995e-08, "apsis"=> 1.4352518e-08, "apsos"=> 3.367064e-09, "apted"=> 2.8648719999999997e-08, "apter"=> 3.55998e-07, "aptly"=> 2.0593580000000002e-06, "aquae"=> 1.0371942000000002e-07, "aquas"=> 3.532004e-08, "araba"=> 7.716378000000001e-08, "araks"=> 6.550119999999999e-09, "arame"=> 2.414466e-08, "arars"=> 6.981938e-09, "arbas"=> 3.826342e-09, "arbor"=> 3.32333e-06, "arced"=> 3.5453639999999996e-07, "archi"=> 1.409082e-07, "arcos"=> 1.470622e-07, "arcus"=> 1.591704e-07, "ardeb"=> 2.908974e-09, "ardor"=> 8.124898e-07, "ardri"=> 2.3207084000000003e-09, "aread"=> 3.453186e-09, "areae"=> 5.7035179999999995e-09, "areal"=> 5.950328e-07, "arear"=> 2.780004e-09, "areas"=> 0.0001157092, "areca"=> 1.0622072e-07, "aredd"=> 1.8442644e-10, "arede"=> 2.28919e-09, "arefy"=> 1.5642344e-10, "areic"=> 1.9631019999999998e-09, "arena"=> 9.709038e-06, "arene"=> 2.0974080000000001e-07, "arepa"=> 1.3851615999999999e-08, "arere"=> 2.977656e-09, "arete"=> 1.20893e-07, "arets"=> 9.530378e-09, "arett"=> 6.487056e-10, "argal"=> 8.891206000000001e-09, "argan"=> 1.1753588e-07, "argil"=> 3.5388740000000002e-09, "argle"=> 9.961014e-09, "argol"=> 8.427671999999999e-09, "argon"=> 1.171182e-06, "argot"=> 1.664722e-07, "argue"=> 3.677706000000001e-05, "argus"=> 8.639306e-07, "arhat"=> 7.819523999999999e-08, "arias"=> 9.067424e-07, "ariel"=> 2.403532e-06, "ariki"=> 6.940162e-08, "arils"=> 3.136586e-08, "ariot"=> 9.75523e-10, "arise"=> 2.2708539999999997e-05, "arish"=> 7.117304e-08, "arked"=> 3.944012e-09, "arled"=> 2.756304e-10, "arles"=> 4.5954080000000004e-07, "armed"=> 3.182364e-05, "armer"=> 6.062769999999999e-08, "armet"=> 7.63685e-09, "armil"=> 7.971388e-10, "armor"=> 7.447496e-06, "arnas"=> 9.170254e-09, "arnut"=> 2.4344354e-10, "aroba"=> 1.2059668e-09, "aroha"=> 8.06293e-08, "aroid"=> 7.036852e-09, "aroma"=> 3.729052e-06, "arose"=> 1.3868720000000001e-05, "arpas"=> 3.7722116e-09, "arpen"=> 1.2881686e-09, "arrah"=> 7.044996e-08, "arras"=> 6.318534e-07, "array"=> 2.08318e-05, "arret"=> 2.3142924e-08, "arris"=> 5.1573039999999994e-08, "arrow"=> 1.4423759999999998e-05, "arroz"=> 1.111882e-07, "arsed"=> 1.0778684e-07, "arses"=> 1.464288e-07, "arsey"=> 7.052654000000001e-09, "arsis"=> 1.58896e-08, "arson"=> 1.302002e-06, "artal"=> 2.533796e-08, "artel"=> 2.9315359999999997e-08, "artic"=> 8.886131999999999e-08, "artis"=> 2.30632e-07, "artsy"=> 2.038054e-07, "aruhe"=> 7.795365999999999e-10, "arums"=> 8.38806e-09, "arval"=> 2.0083200000000003e-08, "arvee"=> 1.8230775799999998e-08, "arvos"=> 7.671658e-10, "aryls"=> 4.360410000000001e-09, "asana"=> 2.295768e-07, "ascon"=> 1.2422e-08, "ascot"=> 2.7119800000000004e-07, "ascus"=> 4.83845e-08, "asdic"=> 4.176348e-08, "ashed"=> 4.3825639999999996e-08, "ashen"=> 1.0155358e-06, "ashes"=> 6.931778e-06, "ashet"=> 1.8622059999999999e-09, "aside"=> 4.260204e-05, "asked"=> 0.000335144, "asker"=> 9.25406e-08, "askew"=> 8.697334e-07, "askoi"=> 2.9680380000000002e-09, "askos"=> 7.999808e-09, "aspen"=> 1.600862e-06, "asper"=> 1.4067140000000002e-07, "aspic"=> 1.1122202e-07, "aspie"=> 4.9462079999999996e-08, "aspis"=> 3.4401799999999995e-08, "aspro"=> 1.7582760000000002e-08, "assai"=> 7.43754e-08, "assam"=> 1.346752e-06, "assay"=> 4.975864e-06, "asses"=> 1.736102e-06, "asset"=> 2.15204e-05, "assez"=> 1.9197240000000002e-07, "assot"=> 3.208086e-10, "aster"=> 7.351616e-07, "astir"=> 2.305676e-07, "astun"=> 6.007076e-10, "asura"=> 2.8988542e-07, "asway"=> 1.970228e-09, "aswim"=> 4.294554000000001e-09, "asyla"=> 5.7525404e-09, "ataps"=> 1.7427354e-09, "ataxy"=> 5.349058e-09, "atigi"=> 1.4135256e-09, "atilt"=> 9.071601999999999e-09, "atimy"=> 1.9463580000000002e-10, "atlas"=> 5.321e-06, "atman"=> 3.6812600000000005e-07, "atmas"=> 7.47743e-10, "atmos"=> 4.1457820000000003e-07, "atocs"=> 5.926967999999999e-11, "atoke"=> 1.1163946000000001e-09, "atoks"=> 7.360788000000001e-11, "atoll"=> 5.15256e-07, "atoms"=> 1.3331619999999998e-05, "atomy"=> 7.0530439999999995e-09, "atone"=> 8.432026000000001e-07, "atony"=> 8.739134000000001e-08, "atopy"=> 1.509906e-07, "atria"=> 6.594342000000001e-07, "atrip"=> 1.1829990000000001e-08, "attap"=> 9.482268e-09, "attar"=> 1.98065e-07, "attic"=> 4.1310660000000005e-06, "atuas"=> 1.1603356e-09, "audad"=> 1.6097076000000001e-10, "audio"=> 1.0169972000000001e-05, "audit"=> 1.413412e-05, "auger"=> 6.68498e-07, "aught"=> 1.4547558e-06, "augur"=> 2.90442e-07, "aulas"=> 1.5309979999999996e-08, "aulic"=> 3.438272e-08, "auloi"=> 6.459562000000001e-09, "aulos"=> 5.4171599999999995e-08, "aumil"=> 1.4605684e-09, "aunes"=> 2.681054e-09, "aunts"=> 2.025526e-06, "aunty"=> 1.340736e-06, "aurae"=> 1.015942e-08, "aural"=> 1.1639422000000002e-06, "aurar"=> 1.8636959999999996e-09, "auras"=> 5.062250000000001e-07, "aurei"=> 3.491552e-08, "aures"=> 4.248876e-08, "auric"=> 1.5105595999999999e-07, "auris"=> 5.579410000000001e-08, "aurum"=> 1.4844999999999998e-07, "autos"=> 3.859536e-07, "auxin"=> 5.47704e-07, "avail"=> 3.830104e-06, "avale"=> 4.163038e-09, "avant"=> 3.6653120000000006e-06, "avast"=> 6.540144e-08, "avels"=> 5.748428400000001e-10, "avens"=> 3.0428100000000005e-08, "avers"=> 2.526018e-07, "avert"=> 1.7037139999999998e-06, "avgas"=> 1.6930800000000004e-08, "avian"=> 1.7512199999999998e-06, "avine"=> 4.856108e-09, "avion"=> 6.144166000000001e-08, "avise"=> 3.4091639999999996e-08, "aviso"=> 5.324202e-08, "avize"=> 1.3848403999999999e-08, "avoid"=> 6.908247999999999e-05, "avows"=> 8.622242e-08, "avyze"=> 0.0, "await"=> 3.7368359999999997e-06, "awake"=> 1.658142e-05, "award"=> 1.825182e-05, "aware"=> 6.428436e-05, "awarn"=> 4.254300400000001e-10, "awash"=> 8.548787999999999e-07, "awato"=> 9.704392e-11, "awave"=> 4.395266e-09, "aways"=> 9.728304e-08, "awdls"=> 0.0, "aweel"=> 3.584452e-08, "aweto"=> 1.1413264e-09, "awful"=> 1.57734e-05, "awing"=> 2.10503e-08, "awmry"=> 6.546153999999999e-10, "awned"=> 2.6460740000000004e-08, "awner"=> 9.080980000000001e-10, "awoke"=> 5.216806e-06, "awols"=> 4.396108000000001e-09, "awork"=> 4.547268e-09, "axels"=> 1.169202e-08, "axial"=> 6.932043999999999e-06, "axile"=> 1.8904276e-08, "axils"=> 9.871616000000001e-08, "axing"=> 3.406578e-08, "axiom"=> 1.9975440000000003e-06, "axion"=> 6.545622e-08, "axite"=> 1.6685790000000001e-09, "axled"=> 2.738682e-09, "axles"=> 4.4695899999999995e-07, "axman"=> 1.7550988e-08, "axmen"=> 9.775142e-09, "axoid"=> 1.4172174e-09, "axone"=> 2.028728e-09, "axons"=> 1.54627e-06, "ayahs"=> 2.6284979999999997e-08, "ayaya"=> 2.8117840000000004e-09, "ayelp"=> 1.4718582e-10, "aygre"=> 1.1895778e-10, "ayins"=> 5.909007999999999e-10, "ayont"=> 1.198672e-08, "ayres"=> 7.429314e-07, "ayrie"=> 2.0396151999999997e-09, "azans"=> 9.336338e-10, "azide"=> 2.898688e-07, "azido"=> 6.093532e-08, "azine"=> 1.822446e-08, "azlon"=> 3.139384e-09, "azoic"=> 1.463738e-08, "azole"=> 1.427214e-07, "azons"=> 2.6246284e-10, "azote"=> 3.536034e-08, "azoth"=> 2.271972e-08, "azuki"=> 2.509434e-08, "azure"=> 3.332478e-06, "azurn"=> 4.573855e-10, "azury"=> 6.322188e-10, "azygy"=> 0.0, "azyme"=> 1.0708686e-09, "azyms"=> 1.2372976e-10, "baaed"=> 4.663358e-09, "baals"=> 1.1117048000000002e-07, "babas"=> 4.978232e-08, "babel"=> 1.5961040000000002e-06, "babes"=> 1.038883e-06, "babka"=> 3.8664180000000006e-08, "baboo"=> 4.6571e-08, "babul"=> 2.3267139999999997e-08, "babus"=> 3.561332e-08, "bacca"=> 8.37371e-08, "bacco"=> 4.465782e-08, "baccy"=> 5.5099260000000006e-08, "bacha"=> 8.009242e-08, "bachs"=> 4.0846019999999994e-08, "backs"=> 8.726041999999999e-06, "bacon"=> 1.072936e-05, "baddy"=> 1.537036e-08, "badge"=> 3.925914e-06, "badly"=> 1.80913e-05, "baels"=> 1.8560628e-09, "baffs"=> 1.9700612e-10, "baffy"=> 8.133833999999999e-09, "bafts"=> 6.29765e-10, "bagel"=> 5.86682e-07, "baggy"=> 1.1074176e-06, "baghs"=> 1.4961583999999998e-09, "bagie"=> 4.6096059999999996e-10, "bahts"=> 6.27397e-09, "bahus"=> 2.3420199999999997e-09, "bahut"=> 3.11654e-08, "bails"=> 8.872964e-08, "bairn"=> 3.3752299999999994e-07, "baisa"=> 1.5195664e-08, "baith"=> 8.671846000000001e-08, "baits"=> 2.751638e-07, "baiza"=> 3.378034e-09, "baize"=> 2.4537559999999996e-07, "bajan"=> 3.98799e-08, "bajra"=> 4.4130640000000004e-08, "bajri"=> 1.9065620000000003e-09, "bajus"=> 2.0037483999999997e-09, "baked"=> 6.470984e-06, "baken"=> 3.741464e-08, "baker"=> 1.3764259999999999e-05, "bakes"=> 2.4488960000000005e-07, "bakra"=> 1.2999768e-08, "balas"=> 1.0283681999999998e-07, "balds"=> 1.179121e-08, "baldy"=> 3.32525e-07, "baled"=> 1.2598457999999998e-07, "baler"=> 8.148155999999998e-08, "bales"=> 1.4244e-06, "balks"=> 9.903374e-08, "balky"=> 5.0539240000000004e-08, "balls"=> 1.357656e-05, "bally"=> 2.622148e-07, "balms"=> 7.777296e-08, "balmy"=> 6.614168e-07, "baloo"=> 1.95497e-07, "balsa"=> 1.352324e-07, "balti"=> 5.516248e-08, "balun"=> 5.214534e-08, "balus"=> 2.2730538000000002e-08, "bambi"=> 3.215612e-07, "banak"=> 1.0405297999999999e-08, "banal"=> 1.2266e-06, "banco"=> 7.667669999999999e-07, "bancs"=> 1.616024e-08, "banda"=> 5.665164e-07, "bandh"=> 2.3551120000000002e-08, "bands"=> 1.3050200000000001e-05, "bandy"=> 2.930556e-07, "baned"=> 2.643494e-09, "banes"=> 1.0588812000000001e-07, "bangs"=> 1.367594e-06, "bania"=> 4.1430300000000004e-08, "banjo"=> 8.343874e-07, "banks"=> 3.681756e-05, "banns"=> 2.309166e-07, "bants"=> 2.2592000000000002e-09, "bantu"=> 8.404431999999999e-07, "banty"=> 1.748038e-08, "banya"=> 9.686200000000001e-08, "bapus"=> 2.4302420000000005e-10, "barbe"=> 1.575896e-07, "barbs"=> 4.897832e-07, "barby"=> 7.252932e-08, "barca"=> 2.693172e-07, "barde"=> 2.556188e-08, "bardo"=> 2.3612580000000002e-07, "bards"=> 4.973786e-07, "bardy"=> 5.03106e-08, "bared"=> 1.7468240000000002e-06, "barer"=> 5.534874e-08, "bares"=> 1.599402e-07, "barfi"=> 1.783382e-08, "barfs"=> 4.742916e-09, "barge"=> 2.179834e-06, "baric"=> 1.9267e-08, "barks"=> 1.058561e-06, "barky"=> 2.582834e-08, "barms"=> 7.407302e-10, "barmy"=> 8.364120000000001e-08, "barns"=> 1.444498e-06, "barny"=> 2.631386e-08, "baron"=> 8.024482e-06, "barps"=> 9.920428e-11, "barra"=> 3.9098339999999997e-07, "barre"=> 7.768372e-07, "barro"=> 2.6463059999999996e-07, "barry"=> 8.93825e-06, "barye"=> 1.1356826e-08, "basal"=> 5.557777999999999e-06, "basan"=> 8.913178e-08, "based"=> 0.00032660539999999997, "basen"=> 1.837084e-08, "baser"=> 3.59805e-07, "bases"=> 1.2148020000000001e-05, "basho"=> 8.680438e-08, "basic"=> 0.0001018077, "basij"=> 7.469820000000001e-08, "basil"=> 6.442776e-06, "basin"=> 1.235644e-05, "basis"=> 9.328449999999999e-05, "basks"=> 5.7570580000000005e-08, "bason"=> 7.40197e-08, "basse"=> 2.104178e-07, "bassi"=> 1.748918e-07, "basso"=> 4.4079419999999995e-07, "bassy"=> 1.4776659999999998e-08, "basta"=> 1.876172e-07, "baste"=> 2.504562e-07, "basti"=> 1.094988e-07, "basto"=> 2.682252e-08, "basts"=> 5.100196e-09, "batch"=> 7.053304e-06, "bated"=> 2.793468e-07, "bates"=> 3.1518e-06, "bathe"=> 2.0439959999999997e-06, "baths"=> 3.2547720000000002e-06, "batik"=> 1.8005359999999998e-07, "baton"=> 2.2593820000000003e-06, "batta"=> 4.141496e-08, "batts"=> 7.889624e-08, "battu"=> 3.0134759999999994e-08, "batty"=> 3.830848e-07, "bauds"=> 4.179330000000001e-09, "bauks"=> 4.089712e-09, "baulk"=> 1.0778528000000002e-07, "baurs"=> 1.1564134e-09, "bavin"=> 1.931176e-08, "bawds"=> 3.875204e-08, "bawdy"=> 4.731542e-07, "bawks"=> 5.583453999999999e-10, "bawls"=> 3.6172439999999996e-08, "bawns"=> 2.583172e-09, "bawrs"=> 6.661892e-11, "bawty"=> 6.812824000000001e-10, "bayed"=> 8.325081999999999e-08, "bayer"=> 1.008481e-06, "bayes"=> 1.4305879999999998e-06, "bayle"=> 4.723076e-07, "bayou"=> 8.809508e-07, "bayts"=> 2.4118254e-09, "bazar"=> 2.413182e-07, "bazoo"=> 4.293236e-09, "beach"=> 3.326728e-05, "beads"=> 5.793735999999999e-06, "beady"=> 5.599028e-07, "beaks"=> 5.940352e-07, "beaky"=> 6.546578e-08, "beals"=> 1.4890079999999998e-07, "beams"=> 7.71596e-06, "beamy"=> 2.107192e-08, "beano"=> 5.587022e-08, "beans"=> 1.1412960000000001e-05, "beany"=> 3.946906e-08, "beard"=> 1.0946438000000001e-05, "beare"=> 3.307608e-07, "bears"=> 1.324306e-05, "beast"=> 1.728846e-05, "beath"=> 2.9154819999999998e-08, "beats"=> 5.693044e-06, "beaty"=> 1.553976e-07, "beaus"=> 5.9580059999999994e-08, "beaut"=> 6.725730000000001e-08, "beaux"=> 7.58637e-07, "bebop"=> 2.0037139999999998e-07, "becap"=> 6.889234e-11, "becke"=> 5.599412000000001e-08, "becks"=> 1.941736e-07, "bedad"=> 5.608278e-08, "bedel"=> 2.8657039999999996e-08, "bedes"=> 1.7403848e-08, "bedew"=> 1.6287079999999998e-08, "bedim"=> 6.089972e-09, "bedye"=> 4.974018e-11, "beech"=> 1.5971160000000002e-06, "beedi"=> 3.3360400000000004e-08, "beefs"=> 3.402378e-08, "beefy"=> 4.6212799999999997e-07, "beeps"=> 4.179446e-07, "beers"=> 3.4442140000000004e-06, "beery"=> 1.3512299999999999e-07, "beets"=> 1.0578608e-06, "befit"=> 1.4903660000000002e-07, "befog"=> 7.771773999999999e-09, "begad"=> 4.3005399999999995e-08, "began"=> 0.000191744, "begar"=> 1.2798032e-08, "begat"=> 1.1297895999999997e-06, "begem"=> 1.1919932e-09, "beget"=> 5.475892000000001e-07, "begin"=> 7.206562e-05, "begot"=> 4.2480900000000003e-07, "begum"=> 3.618822e-07, "begun"=> 2.7279460000000003e-05, "beige"=> 1.5336359999999997e-06, "beigy"=> 1.89857e-09, "being"=> 0.0006041744, "beins"=> 1.4172146e-08, "bekah"=> 9.859668e-08, "belah"=> 3.450742e-08, "belar"=> 1.516448e-08, "belay"=> 2.1689599999999996e-07, "belch"=> 3.4461739999999996e-07, "belee"=> 1.7002579999999999e-09, "belga"=> 1.3381059999999998e-08, "belie"=> 3.6106100000000006e-07, "belle"=> 5.5139380000000005e-06, "bells"=> 6.68092e-06, "belly"=> 1.508164e-05, "belon"=> 3.845072e-08, "below"=> 0.00011265920000000001, "belts"=> 2.983476e-06, "bemad"=> 6.2070944e-10, "bemas"=> 2.0022534e-09, "bemix"=> 8.944666000000001e-11, "bemud"=> 1.3829402e-10, "bench"=> 1.5162700000000001e-05, "bends"=> 2.423982e-06, "bendy"=> 8.89341e-08, "benes"=> 1.1624326e-07, "benet"=> 2.810184e-07, "benga"=> 4.669178e-08, "benis"=> 3.3971716e-08, "benne"=> 1.029197e-07, "benni"=> 4.6216300000000005e-08, "benny"=> 2.701622e-06, "bento"=> 2.317634e-07, "bents"=> 6.134384000000001e-08, "benty"=> 4.766385999999999e-09, "bepat"=> 1.3567798e-10, "beray"=> 1.2593066e-09, "beres"=> 3.452114000000001e-08, "beret"=> 4.968182e-07, "bergs"=> 1.1881084e-07, "berko"=> 3.954364e-08, "berks"=> 1.49167e-07, "berme"=> 2.923708e-09, "berms"=> 8.367034e-08, "berob"=> 7.319206e-11, "berry"=> 5.909408000000001e-06, "berth"=> 1.70556e-06, "beryl"=> 1.0781468e-06, "besat"=> 1.8650696000000002e-08, "besaw"=> 2.585352e-09, "besee"=> 6.170728e-10, "beses"=> 4.836082e-09, "beset"=> 1.768668e-06, "besit"=> 1.3399916e-08, "besom"=> 7.06395e-08, "besot"=> 3.735508e-09, "besti"=> 2.8688200000000004e-09, "bests"=> 7.641839999999999e-08, "betas"=> 2.575788e-07, "beted"=> 1.5936012e-10, "betel"=> 3.8176779999999997e-07, "betes"=> 1.64926e-08, "beths"=> 7.016468e-09, "betid"=> 8.145179999999999e-09, "beton"=> 5.676696e-08, "betta"=> 1.069775e-07, "betty"=> 7.959162000000001e-06, "bevel"=> 5.133253999999999e-07, "bever"=> 1.015375e-07, "bevor"=> 4.701926000000001e-08, "bevue"=> 3.6056849999999996e-10, "bevvy"=> 1.5889482e-08, "bewet"=> 8.415402e-10, "bewig"=> 1.1125824e-09, "bezel"=> 9.882256e-08, "bezes"=> 9.749077999999999e-09, "bezil"=> 1.39637718e-09, "bezzy"=> 1.1867652000000002e-09, "bhais"=> 5.114418e-09, "bhaji"=> 3.925562e-08, "bhang"=> 5.102656e-08, "bhats"=> 5.3880919999999996e-09, "bhels"=> 8.082260000000001e-11, "bhoot"=> 1.0337568e-08, "bhuna"=> 5.5288e-09, "bhuts"=> 2.3843680000000003e-09, "biach"=> 5.970434000000001e-09, "biali"=> 5.6934884e-09, "bialy"=> 1.766484e-08, "bibbs"=> 6.652427400000001e-08, "bibes"=> 6.955676e-09, "bible"=> 0.0001338551, "biccy"=> 1.0579146e-09, "bicep"=> 5.188986e-07, "bices"=> 3.6644079999999997e-10, "biddy"=> 8.525275999999998e-07, "bided"=> 1.62259e-07, "bider"=> 9.67106e-09, "bides"=> 5.495552e-08, "bidet"=> 8.619072e-08, "bidis"=> 1.9459119999999998e-08, "bidon"=> 1.3245456e-08, "bield"=> 9.385762e-09, "biers"=> 4.078748e-08, "biffo"=> 7.346781999999999e-09, "biffs"=> 3.417048e-09, "biffy"=> 1.9102812e-08, "bifid"=> 1.316634e-07, "bigae"=> 1.2916081999999999e-09, "biggs"=> 7.138144000000001e-07, "biggy"=> 2.180202e-08, "bigha"=> 2.546234e-08, "bight"=> 3.6948919999999997e-07, "bigly"=> 1.2228148e-08, "bigos"=> 1.5223219999999997e-08, "bigot"=> 3.3820220000000004e-07, "bijou"=> 2.00823e-07, "biked"=> 9.784405999999999e-08, "biker"=> 1.0274271999999999e-06, "bikes"=> 2.556812e-06, "bikie"=> 2.984684e-08, "bilbo"=> 2.1058659999999998e-07, "bilby"=> 6.806642000000001e-08, "biled"=> 2.0336982e-08, "biles"=> 9.129732e-08, "bilge"=> 3.562198e-07, "bilgy"=> 7.782361999999999e-10, "bilks"=> 6.859057999999999e-09, "bills"=> 1.325334e-05, "billy"=> 1.540694e-05, "bimah"=> 2.695346e-08, "bimas"=> 5.738466e-09, "bimbo"=> 2.51433e-07, "binal"=> 2.663676e-09, "bindi"=> 1.3940019999999998e-07, "binds"=> 4.4400619999999994e-06, "biner"=> 1.3489960000000001e-08, "bines"=> 3.8125239999999996e-08, "binge"=> 1.840394e-06, "bingo"=> 1.353458e-06, "bings"=> 1.5439344e-08, "bingy"=> 4.599702e-09, "binit"=> 5.105495999999999e-09, "binks"=> 9.43832e-08, "bints"=> 3.48106e-09, "biogs"=> 1.2710154e-09, "biome"=> 3.5929e-07, "biont"=> 8.4439e-10, "biota"=> 5.353898e-07, "biped"=> 1.7788320000000002e-07, "bipod"=> 5.7119119999999995e-08, "birch"=> 3.241856e-06, "birds"=> 3.116266e-05, "birks"=> 2.144826e-07, "birle"=> 2.8357253999999995e-08, "birls"=> 1.6476392e-09, "biros"=> 2.3536179999999998e-08, "birrs"=> 1.5808152e-09, "birse"=> 2.103114e-08, "birsy"=> 7.867437999999999e-11, "birth"=> 5.4336300000000004e-05, "bises"=> 3.889648e-09, "bisks"=> 4.863564e-10, "bisom"=> 2.97034e-09, "bison"=> 1.4651920000000002e-06, "bitch"=> 9.424146e-06, "biter"=> 1.415242e-07, "bites"=> 4.235606e-06, "bitos"=> 7.85364e-10, "bitou"=> 6.9452259999999996e-09, "bitsy"=> 2.39004e-07, "bitte"=> 1.819816e-07, "bitts"=> 4.101462e-08, "bitty"=> 2.5086059999999996e-07, "bivia"=> 3.1644800000000006e-09, "bivvy"=> 2.357838e-08, "bizes"=> 1.3614066e-10, "bizzo"=> 3.47808e-09, "bizzy"=> 1.600704e-08, "blabs"=> 1.498214e-08, "black"=> 0.0002237494, "blade"=> 1.6452999999999996e-05, "blads"=> 1.2729744e-09, "blady"=> 5.3085e-09, "blaer"=> 1.1343304e-09, "blaes"=> 8.188122e-09, "blaff"=> 2.023422e-09, "blags"=> 1.5703999999999999e-09, "blahs"=> 1.402652e-08, "blain"=> 2.2868700000000001e-07, "blame"=> 2.339658e-05, "blams"=> 9.420412e-10, "bland"=> 2.967486e-06, "blank"=> 1.8227120000000002e-05, "blare"=> 3.33319e-07, "blart"=> 4.578414e-09, "blase"=> 1.0064106000000001e-07, "blash"=> 6.694902000000001e-09, "blast"=> 9.303026000000002e-06, "blate"=> 8.929454000000001e-09, "blats"=> 3.450028e-09, "blatt"=> 2.416774e-07, "blaud"=> 4.310316e-09, "blawn"=> 4.305804e-09, "blaws"=> 4.7707080000000006e-09, "blays"=> 1.3937984399999995e-08, "blaze"=> 3.989915999999999e-06, "bleak"=> 3.4861260000000003e-06, "blear"=> 5.3319480000000003e-08, "bleat"=> 1.9457080000000002e-07, "blebs"=> 7.64226e-08, "blech"=> 5.70477e-08, "bleed"=> 2.9650459999999996e-06, "bleep"=> 1.384136e-07, "blees"=> 6.040308e-09, "blend"=> 8.713493999999999e-06, "blent"=> 9.907012e-08, "blert"=> 1.0999764600000001e-09, "bless"=> 9.547401999999998e-06, "blest"=> 7.001542e-07, "blets"=> 5.690696000000001e-10, "bleys"=> 1.2575600000000001e-08, "blimp"=> 1.580584e-07, "blimy"=> 4.418348e-09, "blind"=> 2.7438880000000005e-05, "bling"=> 2.850502e-07, "blini"=> 2.819386e-08, "blink"=> 4.804858e-06, "blins"=> 9.404052e-10, "bliny"=> 5.584169999999999e-09, "blips"=> 1.3686680000000002e-07, "bliss"=> 5.904449999999999e-06, "blist"=> 6.563348e-09, "blite"=> 4.0150496e-08, "blits"=> 1.4621239999999998e-08, "blitz"=> 1.1731806e-06, "blive"=> 1.9744912000000002e-07, "bloat"=> 2.02793e-07, "blobs"=> 4.4570699999999996e-07, "block"=> 4.610912e-05, "blocs"=> 6.826546e-07, "blogs"=> 3.26413e-06, "bloke"=> 1.563362e-06, "blond"=> 7.785798e-06, "blood"=> 0.0001649494, "blook"=> 2.0047759999999997e-09, "bloom"=> 7.738566e-06, "bloop"=> 2.717066e-08, "blore"=> 5.0805619999999996e-08, "blots"=> 3.057216e-07, "blown"=> 9.05491e-06, "blows"=> 7.128988000000001e-06, "blowy"=> 2.3290079999999998e-08, "blubs"=> 3.0578220000000003e-09, "blude"=> 1.60862e-08, "bluds"=> 6.540772e-10, "bludy"=> 2.819624e-09, "blued"=> 2.7098459999999997e-07, "bluer"=> 2.0848720000000002e-07, "blues"=> 6.25312e-06, "bluet"=> 2.11331e-08, "bluey"=> 1.2672596e-07, "bluff"=> 3.2540700000000003e-06, "bluid"=> 2.1625e-08, "blume"=> 4.311082e-07, "blunk"=> 1.46537e-08, "blunt"=> 5.669918e-06, "blurb"=> 3.260168e-07, "blurs"=> 6.815704e-07, "blurt"=> 6.641166e-07, "blush"=> 4.896848e-06, "blype"=> 1.0249806000000001e-10, "boabs"=> 1.0542436000000002e-09, "boaks"=> 2.1257966000000004e-09, "board"=> 8.255864000000001e-05, "boars"=> 4.1733439999999997e-07, "boart"=> 5.274538e-09, "boast"=> 3.5578840000000003e-06, "boats"=> 1.3974760000000001e-05, "bobac"=> 5.25644e-10, "bobak"=> 1.7021440000000002e-08, "bobas"=> 7.787872000000001e-10, "bobby"=> 8.626128e-06, "bobol"=> 5.793471999999999e-09, "bobos"=> 2.114716e-08, "bocca"=> 1.871942e-07, "bocce"=> 5.3194819999999996e-08, "bocci"=> 3.3498979999999995e-08, "boche"=> 2.726968e-07, "bocks"=> 2.2654260000000002e-08, "boded"=> 1.9005e-07, "bodes"=> 1.6309840000000002e-07, "bodge"=> 2.312258e-08, "bodhi"=> 4.134486e-07, "bodle"=> 2.063928e-08, "boeps"=> 8.084333999999999e-11, "boets"=> 7.3665740000000006e-09, "boeuf"=> 2.0958779999999995e-07, "boffo"=> 2.586858e-08, "boffs"=> 1.3428086e-09, "bogan"=> 1.1954399999999998e-07, "bogey"=> 2.5618799999999996e-07, "boggy"=> 3.140956e-07, "bogie"=> 2.390914e-07, "bogle"=> 2.4313820000000003e-07, "bogue"=> 1.470692e-07, "bogus"=> 9.611611999999999e-07, "bohea"=> 2.7196759999999998e-08, "bohos"=> 4.012358000000001e-09, "boils"=> 1.395012e-06, "boing"=> 8.260743999999999e-08, "boink"=> 1.932468e-08, "boite"=> 1.87228e-08, "boked"=> 6.534858e-10, "bokeh"=> 3.630974000000001e-08, "bokes"=> 3.257982e-08, "bokos"=> 2.7118278000000003e-09, "bolar"=> 2.726112e-08, "bolas"=> 6.858150000000001e-08, "bolds"=> 1.589822e-08, "boles"=> 2.5761480000000004e-07, "bolix"=> 1.3577096000000002e-10, "bolls"=> 6.082412e-08, "bolos"=> 4.602304e-08, "bolts"=> 3.690316e-06, "bolus"=> 1.4099439999999998e-06, "bomas"=> 1.507436e-08, "bombe"=> 7.02528e-08, "bombo"=> 2.245044e-08, "bombs"=> 6.273274e-06, "bonce"=> 1.3042052e-08, "bonds"=> 2.2105999999999998e-05, "boned"=> 6.095001999999999e-07, "boner"=> 2.4645939999999994e-07, "bones"=> 2.5214899999999995e-05, "boney"=> 2.3346680000000002e-07, "bongo"=> 2.399334e-07, "bongs"=> 3.8370700000000004e-08, "bonie"=> 1.643849e-08, "bonks"=> 8.866596e-09, "bonne"=> 5.629584e-07, "bonny"=> 7.787352e-07, "bonus"=> 5.6314899999999995e-06, "bonza"=> 1.0561823999999999e-08, "bonze"=> 2.554512e-08, "booai"=> 1.2605616e-10, "booay"=> 9.763145999999999e-11, "boobs"=> 9.916511999999998e-07, "booby"=> 6.266299999999999e-07, "boody"=> 1.8917040000000003e-08, "booed"=> 2.2446680000000004e-07, "boofy"=> 3.6159200000000005e-09, "boogy"=> 3.234842e-09, "boohs"=> 3.7803700000000005e-10, "books"=> 0.000113126, "booky"=> 9.579492e-09, "bools"=> 8.464158e-09, "booms"=> 8.820051999999999e-07, "boomy"=> 8.207936e-09, "boong"=> 6.75758e-09, "boons"=> 2.3396859999999997e-07, "boord"=> 2.293132e-08, "boors"=> 6.639216e-08, "boose"=> 4.12752e-08, "boost"=> 7.096163999999999e-06, "booth"=> 8.216102e-06, "boots"=> 1.8352860000000002e-05, "booty"=> 1.553676e-06, "booze"=> 1.9134640000000003e-06, "boozy"=> 1.801422e-07, "boppy"=> 6.319409999999999e-09, "borak"=> 3.46641e-08, "boral"=> 2.026856e-08, "boras"=> 2.11829e-08, "borax"=> 2.298498e-07, "borde"=> 1.2944862e-07, "bords"=> 9.937902e-08, "bored"=> 8.293962000000002e-06, "boree"=> 4.708792e-09, "borel"=> 5.39884e-07, "borer"=> 3.86444e-07, "bores"=> 4.544404e-07, "borgo"=> 2.279518e-07, "boric"=> 2.097458e-07, "borks"=> 9.040867999999999e-10, "borms"=> 3.837514e-09, "borna"=> 5.205694e-08, "borne"=> 1.0131652e-05, "boron"=> 1.67852e-06, "borts"=> 4.9852639999999994e-09, "borty"=> 5.032738e-10, "bortz"=> 3.331132e-08, "bosie"=> 3.950768e-08, "bosks"=> 1.2280219999999999e-09, "bosky"=> 4.7081940000000005e-08, "bosom"=> 5.54503e-06, "boson"=> 5.96586e-07, "bossy"=> 9.081308000000001e-07, "bosun"=> 1.725322e-07, "botas"=> 3.1706720000000004e-08, "botch"=> 1.103935e-07, "botel"=> 6.831618e-09, "botes"=> 2.401716e-08, "bothy"=> 1.0616459999999999e-07, "botte"=> 4.921272e-08, "botts"=> 6.680628e-08, "botty"=> 7.819736e-09, "bouge"=> 1.844318e-08, "bough"=> 9.064482000000001e-07, "bouks"=> 1.0698532e-09, "boule"=> 1.950836e-07, "boult"=> 7.336796e-08, "bound"=> 4.544912e-05, "bouns"=> 3.649536e-10, "bourd"=> 3.697096e-09, "bourg"=> 1.85947e-07, "bourn"=> 9.716056e-08, "bouse"=> 1.644566e-08, "bousy"=> 4.201478e-10, "bouts"=> 1.345958e-06, "bovid"=> 2.181082e-08, "bowat"=> 4.424838e-10, "bowed"=> 1.1786484e-05, "bowel"=> 7.188446e-06, "bower"=> 1.496944e-06, "bowes"=> 3.1834740000000003e-07, "bowet"=> 2.6696560000000004e-09, "bowie"=> 2.2461079999999998e-06, "bowls"=> 4.643381999999999e-06, "bowne"=> 5.0539839999999997e-08, "bowrs"=> 9.252572000000001e-10, "bowse"=> 1.744346e-08, "boxed"=> 1.325502e-06, "boxen"=> 6.2925640000000005e-09, "boxer"=> 2.3675800000000003e-06, "boxes"=> 1.653544e-05, "boxla"=> 3.1249706000000004e-10, "boxty"=> 8.19463e-09, "boyar"=> 1.0379706000000002e-07, "boyau"=> 1.791475e-08, "boyed"=> 3.126674e-09, "boyfs"=> 1.1468300000000001e-10, "boygs"=> 9.05258e-11, "boyla"=> 4.178174e-10, "boyos"=> 1.0911338000000001e-08, "boysy"=> 1.304614e-09, "bozos"=> 4.59767e-08, "braai"=> 8.59967e-08, "brace"=> 4.0240019999999996e-06, "brach"=> 1.5637826e-07, "brack"=> 2.350488e-07, "bract"=> 9.079296e-08, "brads"=> 3.0707739999999996e-08, "braes"=> 7.519912e-08, "brags"=> 1.4508739999999998e-07, "braid"=> 2.083498e-06, "brail"=> 1.9448940000000002e-08, "brain"=> 9.165586000000001e-05, "brake"=> 4.552945999999999e-06, "braks"=> 2.148342e-09, "braky"=> 5.535399999999999e-10, "brame"=> 5.7038779999999995e-08, "brand"=> 2.8957720000000004e-05, "brane"=> 1.951918e-07, "brank"=> 1.739462e-08, "brans"=> 5.285688e-08, "brant"=> 9.517088e-07, "brash"=> 7.95547e-07, "brass"=> 8.469142e-06, "brast"=> 3.6679859999999997e-08, "brats"=> 4.46353e-07, "brava"=> 1.569416e-07, "brave"=> 1.560208e-05, "bravi"=> 2.6787080000000003e-08, "bravo"=> 1.679602e-06, "brawl"=> 8.159418000000001e-07, "brawn"=> 3.612954e-07, "braws"=> 3.8470340000000006e-09, "braxy"=> 6.116175999999999e-09, "brays"=> 4.4004800000000005e-08, "braza"=> 8.535376e-09, "braze"=> 5.99057e-08, "bread"=> 3.4327860000000004e-05, "break"=> 7.061344e-05, "bream"=> 3.673696e-07, "brede"=> 1.2750442000000002e-07, "breds"=> 1.1992758e-08, "breed"=> 6.763823999999999e-06, "breem"=> 8.150938000000001e-09, "breer"=> 2.578638e-08, "brees"=> 3.606804e-08, "breid"=> 9.133848000000002e-09, "breis"=> 4.647368e-09, "breme"=> 1.1693798e-08, "brens"=> 8.64595e-09, "brent"=> 3.325724e-06, "brere"=> 3.398122e-09, "brers"=> 3.7853766000000004e-10, "breve"=> 3.0768760000000005e-07, "brews"=> 3.6828599999999997e-07, "breys"=> 1.034292e-10, "briar"=> 7.956892e-07, "bribe"=> 2.237022e-06, "brick"=> 1.2341920000000001e-05, "bride"=> 1.18994e-05, "brief"=> 4.554892000000001e-05, "brier"=> 2.908062e-07, "bries"=> 5.399134e-09, "brigs"=> 1.0314714e-07, "briki"=> 4.038211999999999e-09, "briks"=> 5.856429999999999e-09, "brill"=> 4.397207999999999e-06, "brims"=> 1.600168e-07, "brine"=> 1.660144e-06, "bring"=> 0.0001245252, "brink"=> 3.4653339999999998e-06, "brins"=> 4.641314e-09, "briny"=> 2.317686e-07, "brios"=> 8.3269838e-09, "brise"=> 5.802204000000001e-08, "brisk"=> 2.858436e-06, "briss"=> 6.448672e-08, "brith"=> 4.590626e-08, "brits"=> 5.149302e-07, "britt"=> 9.676093999999999e-07, "brize"=> 2.8110240000000002e-08, "broad"=> 4.7157679999999996e-05, "broch"=> 1.621474e-07, "brock"=> 3.1019719999999998e-06, "brods"=> 2.1043870000000003e-09, "brogh"=> 3.453928e-10, "brogs"=> 5.815834e-10, "broil"=> 3.591408e-07, "broke"=> 4.3705379999999995e-05, "brome"=> 1.491654e-07, "bromo"=> 2.496178e-07, "bronc"=> 1.2302173999999998e-07, "brond"=> 1.6721958e-08, "brood"=> 2.092954e-06, "brook"=> 5.272108e-06, "brool"=> 3.261278e-09, "broom"=> 2.5284920000000002e-06, "broos"=> 2.8829219999999997e-08, "brose"=> 7.872388000000001e-08, "brosy"=> 3.7475546e-09, "broth"=> 4.184312000000001e-06, "brown"=> 7.977228e-05, "brows"=> 6.351082e-06, "brugh"=> 2.310724e-08, "bruin"=> 3.5692859999999996e-07, "bruit"=> 2.335874e-07, "brule"=> 8.556628000000001e-08, "brume"=> 3.14089e-08, "brung"=> 1.2160247999999997e-07, "brunt"=> 1.457106e-06, "brush"=> 1.49058e-05, "brusk"=> 9.421158000000001e-09, "brust"=> 9.314886e-08, "brute"=> 4.222728e-06, "bruts"=> 5.6231459999999995e-09, "buats"=> 9.567364e-11, "buaze"=> 3.0471016e-10, "bubal"=> 1.2474178e-09, "bubas"=> 1.2783170000000002e-08, "bubba"=> 5.141711999999999e-07, "bubbe"=> 6.525836e-08, "bubby"=> 4.38824e-08, "bubus"=> 4.108116e-09, "buchu"=> 2.094798e-08, "bucko"=> 5.251988e-08, "bucks"=> 2.842068e-06, "bucku"=> 2.516022e-10, "budas"=> 1.785214e-09, "buddy"=> 6.332234e-06, "budge"=> 2.018644e-06, "budis"=> 1.2044688e-09, "budos"=> 8.490363999999999e-10, "buffa"=> 7.115524e-08, "buffe"=> 6.816764e-09, "buffi"=> 4.782912e-09, "buffo"=> 3.276038e-08, "buffs"=> 2.901436e-07, "buffy"=> 1.0757186e-06, "bufos"=> 1.3227032e-09, "bufty"=> 2.2545686000000004e-10, "buggy"=> 2.064934e-06, "bugle"=> 7.052522e-07, "buhls"=> 6.245481999999999e-10, "buhrs"=> 1.1727994000000001e-09, "buiks"=> 5.8033000000000004e-09, "build"=> 5.930362e-05, "built"=> 7.782811999999998e-05, "buist"=> 5.6753259999999996e-08, "bukes"=> 2.4300680000000003e-09, "bulbs"=> 2.17642e-06, "bulge"=> 2.048256e-06, "bulgy"=> 3.98972e-08, "bulks"=> 1.06408e-07, "bulky"=> 2.311198e-06, "bulla"=> 2.4533600000000005e-07, "bulls"=> 2.463802e-06, "bully"=> 3.503742e-06, "bulse"=> 7.781729999999999e-10, "bumbo"=> 1.0405359999999998e-08, "bumfs"=> 1.1638442000000001e-10, "bumph"=> 4.924226e-09, "bumps"=> 2.4917599999999998e-06, "bumpy"=> 9.567688e-07, "bunas"=> 7.591304000000001e-10, "bunce"=> 2.84366e-07, "bunch"=> 1.257288e-05, "bunco"=> 4.295184e-08, "bunde"=> 3.445566e-08, "bundh"=> 3.0973864e-09, "bunds"=> 7.592994e-08, "bundt"=> 1.2196956000000002e-07, "bundu"=> 2.0892900000000002e-08, "bundy"=> 7.781629999999998e-07, "bungs"=> 3.6511439999999996e-08, "bungy"=> 2.425812e-08, "bunia"=> 2.5359300000000002e-08, "bunje"=> 2.27613e-09, "bunjy"=> 9.8381238e-10, "bunko"=> 6.274134000000001e-08, "bunks"=> 6.693612e-07, "bunns"=> 8.706742e-09, "bunny"=> 2.5151119999999998e-06, "bunts"=> 3.242862e-08, "bunty"=> 1.729166e-07, "bunya"=> 2.1398899999999996e-08, "buoys"=> 4.719466e-07, "buppy"=> 7.887712e-10, "buran"=> 2.7934779999999998e-08, "buras"=> 2.73101e-08, "burbs"=> 5.5587040000000006e-08, "burds"=> 1.1420772e-08, "buret"=> 4.13495e-08, "burfi"=> 8.384636e-09, "burgh"=> 4.891078e-07, "burgs"=> 1.913956e-08, "burin"=> 8.650302e-08, "burka"=> 9.936550000000002e-08, "burke"=> 7.649668e-06, "burks"=> 1.747348e-07, "burls"=> 1.864026e-08, "burly"=> 1.516738e-06, "burns"=> 1.095266e-05, "burnt"=> 9.826844e-06, "buroo"=> 6.103496e-10, "burps"=> 7.152802e-08, "burqa"=> 1.671716e-07, "burro"=> 2.307436e-07, "burrs"=> 2.0920639999999997e-07, "burry"=> 6.585412000000001e-08, "bursa"=> 6.122081999999999e-07, "burse"=> 1.9856620000000002e-08, "burst"=> 2.3667920000000002e-05, "busby"=> 4.4032259999999997e-07, "bused"=> 7.954454e-08, "buses"=> 5.366358e-06, "bushy"=> 1.4981499999999998e-06, "busks"=> 8.507384e-09, "busky"=> 2.0261480000000003e-09, "bussu"=> 3.578982e-09, "busti"=> 1.0847138e-08, "busts"=> 7.341094e-07, "busty"=> 1.219636e-07, "butch"=> 1.3621539999999999e-06, "buteo"=> 6.135902e-08, "butes"=> 1.707008e-08, "butle"=> 2.8535444e-09, "butoh"=> 8.576217999999999e-08, "butte"=> 7.771360000000001e-07, "butts"=> 1.4181100000000003e-06, "butty"=> 5.646311999999999e-08, "butut"=> 1.1008680000000001e-09, "butyl"=> 8.263996e-07, "buxom"=> 3.45187e-07, "buyer"=> 1.0457719999999998e-05, "buzzy"=> 1.787002e-07, "bwana"=> 1.2094156000000002e-07, "bwazi"=> 5.524928e-11, "byded"=> 4.354506e-11, "bydes"=> 2.9025504e-10, "byked"=> 0.0, "bykes"=> 2.661327e-09, "bylaw"=> 1.694926e-07, "byres"=> 7.688404e-08, "byrls"=> 0.0, "byssi"=> 3.639253e-10, "bytes"=> 1.89596e-06, "byway"=> 1.393072e-07, "caaed"=> 1.6108082e-10, "cabal"=> 4.470776e-07, "cabas"=> 5.627857999999999e-09, "cabby"=> 1.4229400000000002e-07, "caber"=> 3.191212e-08, "cabin"=> 1.834018e-05, "cable"=> 1.29092e-05, "cabob"=> 1.2823319999999999e-09, "caboc"=> 5.086462e-10, "cabre"=> 9.75453e-09, "cacao"=> 9.789628e-07, "cacas"=> 4.282078e-09, "cache"=> 3.6410639999999996e-06, "cacks"=> 1.445576e-09, "cacky"=> 1.2426848e-09, "cacti"=> 4.0051120000000004e-07, "caddy"=> 6.535296e-07, "cadee"=> 4.4334139999999995e-09, "cades"=> 1.783418e-07, "cadet"=> 1.263582e-06, "cadge"=> 6.308926000000001e-08, "cadgy"=> 5.271475999999999e-10, "cadie"=> 6.848448000000001e-08, "cadis"=> 3.299168e-08, "cadre"=> 1.5252939999999999e-06, "caeca"=> 4.685051999999999e-08, "caese"=> 2.6388544e-10, "cafes"=> 7.595094e-07, "caffs"=> 2.4271320000000003e-09, "caged"=> 1.209752e-06, "cager"=> 1.688645e-08, "cages"=> 2.01472e-06, "cagey"=> 2.139054e-07, "cagot"=> 7.061081999999999e-09, "cahow"=> 5.46894e-09, "caids"=> 6.9284959999999985e-09, "cains"=> 2.484368e-08, "caird"=> 2.549678e-07, "cairn"=> 6.375736e-07, "cajon"=> 6.080656e-08, "cajun"=> 5.494627999999999e-07, "caked"=> 7.396178000000001e-07, "cakes"=> 5.497802e-06, "cakey"=> 2.8993020000000003e-08, "calfs"=> 5.102752e-09, "calid"=> 3.458694e-09, "calif"=> 1.350312e-07, "calix"=> 1.7440258000000002e-07, "calks"=> 5.4181700000000005e-09, "calla"=> 4.05383e-07, "calls"=> 5.0075900000000006e-05, "calms"=> 6.82319e-07, "calmy"=> 5.73005e-09, "calos"=> 5.452136e-09, "calpa"=> 1.7188108e-09, "calps"=> 1.5495368000000002e-09, "calve"=> 8.832112e-08, "calyx"=> 4.795254e-07, "caman"=> 3.559356e-09, "camas"=> 8.879298e-08, "camel"=> 3.496098e-06, "cameo"=> 6.25582e-07, "cames"=> 1.4569119999999998e-08, "camis"=> 7.055533999999999e-09, "camos"=> 2.0935220000000002e-08, "campi"=> 1.6312099999999997e-07, "campo"=> 1.31308e-06, "camps"=> 1.093556e-05, "campy"=> 1.360042e-07, "camus"=> 1.0336494e-06, "canal"=> 1.3947960000000002e-05, "candy"=> 7.74659e-06, "caned"=> 1.0038782e-07, "caneh"=> 1.0178634000000001e-10, "caner"=> 4.033492e-08, "canes"=> 8.030453999999999e-07, "cangs"=> 1.6110701999999999e-10, "canid"=> 5.794027999999999e-08, "canna"=> 5.390558e-07, "canns"=> 1.662624e-09, "canny"=> 6.522606e-07, "canoe"=> 4.438074e-06, "canon"=> 7.471116000000001e-06, "canso"=> 4.746538e-08, "canst"=> 1.253957e-06, "canto"=> 1.23564e-06, "cants"=> 3.0784140000000004e-08, "canty"=> 1.5018476e-07, "capas"=> 2.8689159999999998e-08, "caped"=> 1.0718444e-07, "caper"=> 4.5103539999999993e-07, "capes"=> 6.150580000000001e-07, "capex"=> 2.031238e-07, "caphs"=> 2.4783426e-10, "capiz"=> 8.426164000000001e-09, "caple"=> 3.712850000000001e-08, "capon"=> 1.946102e-07, "capos"=> 3.411384e-08, "capot"=> 1.2154206e-08, "capri"=> 5.239815999999999e-07, "capul"=> 4.1575276e-08, "caput"=> 3.285122e-07, "carap"=> 4.894897999999999e-10, "carat"=> 3.05492e-07, "carbo"=> 2.013378e-07, "carbs"=> 1.465168e-06, "carby"=> 6.508354e-08, "cardi"=> 6.695627999999999e-08, "cards"=> 2.2513359999999998e-05, "cardy"=> 5.4001820000000005e-08, "cared"=> 1.5092140000000002e-05, "carer"=> 1.124472e-06, "cares"=> 7.501344e-06, "caret"=> 1.1074474e-07, "carex"=> 1.1017486000000001e-07, "cargo"=> 9.42218e-06, "carks"=> 1.0016342e-09, "carle"=> 1.8866100000000002e-07, "carls"=> 4.2461100000000004e-08, "carns"=> 1.2216340000000001e-08, "carny"=> 5.590018e-08, "carob"=> 1.813258e-07, "carol"=> 8.942160000000001e-06, "carom"=> 3.878854e-08, "caron"=> 5.110646e-07, "carpi"=> 4.0683379999999996e-07, "carps"=> 6.226921999999999e-08, "carrs"=> 2.5839599999999997e-08, "carry"=> 5.760598e-05, "carse"=> 7.129859999999999e-08, "carta"=> 1.0969881999999998e-06, "carte"=> 1.0009929999999998e-06, "carts"=> 2.748686e-06, "carve"=> 2.00447e-06, "carvy"=> 4.065616e-10, "casas"=> 9.093124e-07, "casco"=> 1.142478e-07, "cased"=> 2.762362e-07, "cases"=> 0.00015391900000000003, "casks"=> 6.83888e-07, "casky"=> 9.162310000000001e-10, "caste"=> 6.245446e-06, "casts"=> 3.341184e-06, "casus"=> 1.5206760000000001e-07, "catch"=> 4.1496959999999996e-05, "cater"=> 1.7490899999999998e-06, "cates"=> 2.8216759999999997e-07, "catty"=> 2.239516e-07, "cauda"=> 2.990932e-07, "cauks"=> 1.3626143999999998e-10, "cauld"=> 6.758506e-08, "caulk"=> 1.0872316e-07, "cauls"=> 1.1321562e-08, "caums"=> 6.18005e-11, "caups"=> 4.742768e-10, "cauri"=> 2.080526e-09, "causa"=> 9.517424e-07, "cause"=> 0.0001299014, "cavas"=> 1.3847199999999999e-08, "caved"=> 7.323500000000001e-07, "cavel"=> 5.041582e-09, "caver"=> 2.890432e-08, "caves"=> 4.793202000000001e-06, "cavie"=> 1.135075e-09, "cavil"=> 1.3323495999999998e-07, "cawed"=> 1.2572780000000001e-07, "cawks"=> 4.88967e-11, "caxon"=> 1.4411896e-08, "cease"=> 9.09453e-06, "ceaze"=> 1.0946174e-09, "cebid"=> 5.79705e-10, "cecal"=> 1.3204020000000002e-07, "cecum"=> 3.073548e-07, "cedar"=> 3.765026e-06, "ceded"=> 9.865294000000001e-07, "ceder"=> 3.9099400000000007e-08, "cedes"=> 9.462629999999999e-08, "cedis"=> 2.990456e-08, "ceiba"=> 9.035392e-08, "ceili"=> 8.8405e-09, "ceils"=> 2.1581960000000002e-09, "celeb"=> 7.800506e-08, "cella"=> 2.1691960000000002e-07, "celli"=> 6.344476000000001e-08, "cello"=> 9.323986e-07, "cells"=> 0.00010241344, "celom"=> 1.8201832000000002e-08, "celts"=> 6.179268e-07, "cense"=> 3.9113120000000006e-08, "cento"=> 2.3118400000000004e-07, "cents"=> 5.0625860000000005e-06, "centu"=> 1.508278e-08, "ceorl"=> 1.2370984000000002e-08, "cepes"=> 8.076284e-09, "cerci"=> 2.5514359999999998e-08, "cered"=> 3.8077680000000006e-09, "ceres"=> 7.911170000000001e-07, "cerge"=> 2.862444e-09, "ceria"=> 1.701646e-07, "ceric"=> 3.206304e-08, "cerne"=> 3.937828e-08, "ceroc"=> 9.564254e-10, "ceros"=> 1.282768e-08, "certs"=> 5.993998e-08, "certy"=> 1.911444e-09, "cesse"=> 5.34862e-08, "cesta"=> 6.799078e-08, "cesti"=> 1.1424559999999998e-08, "cetes"=> 4.6574120000000006e-09, "cetyl"=> 4.423952e-08, "cezve"=> 1.8258984e-09, "chace"=> 1.9147019999999996e-07, "chack"=> 2.443336e-08, "chaco"=> 4.952722e-07, "chado"=> 8.043300000000001e-09, "chads"=> 4.357732e-08, "chafe"=> 3.5859320000000005e-07, "chaff"=> 9.524575999999999e-07, "chaft"=> 2.2833439999999997e-09, "chain"=> 4.466296e-05, "chair"=> 6.993162000000001e-05, "chais"=> 1.2306504000000001e-08, "chalk"=> 3.3783640000000004e-06, "chals"=> 3.789952e-09, "champ"=> 1.042516e-06, "chams"=> 3.106906e-08, "chana"=> 1.707956e-07, "chang"=> 6.698932e-06, "chank"=> 1.580652e-08, "chant"=> 3.126336e-06, "chaos"=> 1.3899600000000002e-05, "chape"=> 2.5919219999999996e-08, "chaps"=> 1.900638e-06, "chapt"=> 5.4881879999999996e-08, "chara"=> 1.1264559999999999e-07, "chard"=> 6.527996e-07, "chare"=> 2.519114e-08, "chark"=> 1.0195428e-08, "charm"=> 1.0215244e-05, "charr"=> 3.212398e-08, "chars"=> 1.4093316e-07, "chart"=> 1.76125e-05, "chary"=> 1.525702e-07, "chase"=> 1.6121660000000002e-05, "chasm"=> 1.894996e-06, "chats"=> 8.38393e-07, "chave"=> 5.63416e-08, "chavs"=> 3.5094960000000005e-08, "chawk"=> 5.766636e-09, "chaws"=> 1.1279766e-08, "chaya"=> 1.2206292e-07, "chays"=> 1.7297068e-09, "cheap"=> 1.40973e-05, "cheat"=> 3.1958540000000006e-06, "check"=> 6.895206e-05, "cheek"=> 2.5635699999999996e-05, "cheep"=> 6.599192e-08, "cheer"=> 6.056648e-06, "chefs"=> 1.561532e-06, "cheka"=> 1.6092380000000003e-07, "chela"=> 1.4870480000000002e-07, "chelp"=> 9.575901999999999e-10, "chemo"=> 1.0411331999999998e-06, "chems"=> 2.5923799999999997e-08, "chere"=> 9.365188000000001e-08, "chert"=> 2.796998e-07, "chess"=> 4.856594e-06, "chest"=> 6.941442e-05, "cheth"=> 1.1018752e-08, "chevy"=> 9.389458e-07, "chews"=> 4.231138e-07, "chewy"=> 4.181714e-07, "chiao"=> 1.6118439999999996e-07, "chias"=> 5.65678e-09, "chibs"=> 1.8205624000000001e-09, "chica"=> 2.699116e-07, "chich"=> 1.5639599999999998e-08, "chick"=> 3.0698760000000002e-06, "chico"=> 7.865666e-07, "chics"=> 4.774777999999999e-09, "chide"=> 3.195167999999999e-07, "chief"=> 6.500580000000001e-05, "chiel"=> 3.711648e-08, "chiks"=> 1.1290191999999998e-09, "child"=> 0.00021079140000000002, "chile"=> 7.713812e-06, "chili"=> 2.835496e-06, "chill"=> 9.243887999999999e-06, "chimb"=> 4.636360000000001e-10, "chime"=> 1.042852e-06, "chimo"=> 1.9963760000000002e-08, "chimp"=> 3.109708e-07, "china"=> 0.00010549067999999999, "chine"=> 3.3811160000000004e-07, "ching"=> 1.3733100000000001e-06, "chink"=> 6.34621e-07, "chino"=> 2.7581940000000004e-07, "chins"=> 4.93591e-07, "chips"=> 7.924877999999999e-06, "chirk"=> 3.4493500000000004e-08, "chirl"=> 2.5893948e-10, "chirm"=> 1.0810185999999998e-09, "chiro"=> 3.676126e-08, "chirp"=> 6.268212e-07, "chirr"=> 9.617408e-09, "chirt"=> 1.3975897999999999e-09, "chiru"=> 1.8028460000000002e-08, "chits"=> 8.466379999999999e-08, "chive"=> 1.2492800000000002e-07, "chivs"=> 1.3726626e-09, "chivy"=> 6.96859e-09, "chizz"=> 2.0596799999999997e-09, "chock"=> 3.798818e-07, "choco"=> 1.0228926e-07, "chocs"=> 1.875166e-08, "chode"=> 1.0963288e-08, "chogs"=> 2.438878e-10, "choil"=> 2.2614138e-09, "choir"=> 4.489952e-06, "choke"=> 2.8815640000000004e-06, "choko"=> 1.3874412e-08, "choky"=> 1.902428e-08, "chola"=> 1.433248e-07, "choli"=> 2.336268e-08, "cholo"=> 6.479616e-08, "chomp"=> 1.388524e-07, "chons"=> 2.618512e-09, "choof"=> 2.469996e-09, "chook"=> 1.0054505999999998e-07, "choom"=> 6.625824000000001e-09, "choon"=> 6.745812e-08, "chops"=> 1.6126979999999998e-06, "chord"=> 3.783996e-06, "chore"=> 1.130718e-06, "chose"=> 2.6667420000000003e-05, "chota"=> 8.982902e-08, "chott"=> 1.2429414e-08, "chout"=> 7.676672000000001e-09, "choux"=> 1.0271649999999999e-07, "chowk"=> 1.0500503999999999e-07, "chows"=> 2.432722e-08, "chubs"=> 6.466892e-08, "chuck"=> 4.8681340000000005e-06, "chufa"=> 9.769582000000002e-09, "chuff"=> 6.93797e-08, "chugs"=> 5.843046e-08, "chump"=> 2.366464e-07, "chums"=> 3.417954e-07, "chunk"=> 2.9450000000000006e-06, "churl"=> 1.1373407999999999e-07, "churn"=> 1.1890440000000001e-06, "churr"=> 1.2968338000000001e-08, "chuse"=> 2.198964e-07, "chute"=> 1.157864e-06, "chuts"=> 5.81016e-10, "chyle"=> 6.920518e-08, "chyme"=> 1.46178e-07, "chynd"=> 8.420136e-11, "cibol"=> 2.1942868e-10, "cided"=> 2.809996e-08, "cider"=> 2.59036e-06, "cides"=> 2.1913339999999998e-08, "ciels"=> 2.6153080000000003e-09, "cigar"=> 4.55944e-06, "ciggy"=> 1.4134094000000001e-08, "cilia"=> 6.54649e-07, "cills"=> 4.7715299999999995e-09, "cimar"=> 5.832556e-09, "cimex"=> 4.056192e-08, "cinch"=> 3.4101939999999996e-07, "cinct"=> 2.111278e-09, "cines"=> 2.9682160000000002e-08, "cinqs"=> 2.091786e-10, "cions"=> 5.0618499999999994e-09, "cippi"=> 1.0608170000000001e-08, "circa"=> 2.12475e-06, "circs"=> 9.736014e-09, "cires"=> 7.586002e-09, "cirls"=> 4.737264e-10, "cirri"=> 6.882834e-08, "cisco"=> 1.618496e-06, "cissy"=> 4.6032880000000003e-07, "cists"=> 4.169742e-08, "cital"=> 1.4336296e-09, "cited"=> 2.36693e-05, "citer"=> 1.442196e-08, "cites"=> 4.1688979999999995e-06, "cives"=> 5.636888e-08, "civet"=> 1.3487619999999998e-07, "civic"=> 1.198484e-05, "civie"=> 4.721272399999999e-09, "civil"=> 8.055338e-05, "civvy"=> 3.5611800000000004e-08, "clach"=> 8.929536e-09, "clack"=> 5.727301999999999e-07, "clade"=> 5.236672e-07, "clads"=> 2.402434e-08, "claes"=> 2.652878e-07, "clags"=> 3.065824e-09, "claim"=> 7.156122e-05, "clame"=> 2.718768e-09, "clamp"=> 2.025626e-06, "clams"=> 1.0030512e-06, "clang"=> 1.0847652e-06, "clank"=> 4.5084439999999995e-07, "clans"=> 2.52543e-06, "claps"=> 6.278557999999999e-07, "clapt"=> 2.2899760000000003e-08, "claro"=> 2.17788e-07, "clart"=> 1.1663486000000001e-08, "clary"=> 5.093536e-07, "clash"=> 4.762404e-06, "clasp"=> 2.064438e-06, "class"=> 0.000162074, "clast"=> 6.556176e-08, "clats"=> 7.100172000000001e-10, "claut"=> 1.560959e-09, "clave"=> 2.2056e-07, "clavi"=> 1.1432418e-08, "claws"=> 5.176052e-06, "clays"=> 1.1819839999999998e-06, "clean"=> 5.083136e-05, "clear"=> 0.000166256, "cleat"=> 1.59449e-07, "cleck"=> 2.44365e-09, "cleek"=> 2.977301e-07, "cleep"=> 1.3134646e-09, "clefs"=> 5.8255140000000006e-08, "cleft"=> 2.960096e-06, "clegs"=> 3.0333220000000003e-09, "cleik"=> 1.0974672e-09, "clems"=> 2.3382954e-09, "clepe"=> 9.198916000000002e-09, "clept"=> 3.705646e-09, "clerk"=> 1.1216038000000001e-05, "cleve"=> 4.2898e-07, "clews"=> 8.084190000000001e-08, "click"=> 3.702342e-05, "clied"=> 2.7069546e-10, "clies"=> 2.4373099999999997e-09, "cliff"=> 1.0030010000000001e-05, "clift"=> 2.473836e-07, "climb"=> 1.5098759999999999e-05, "clime"=> 2.978632e-07, "cline"=> 6.779992000000001e-07, "cling"=> 3.4261119999999996e-06, "clink"=> 7.787540000000001e-07, "clint"=> 1.9993e-06, "clipe"=> 2.0112758e-09, "clips"=> 2.783718e-06, "clipt"=> 1.64436e-08, "clits"=> 1.9301939999999997e-08, "cloak"=> 7.969264e-06, "cloam"=> 2.1759014e-09, "clock"=> 2.0318460000000003e-05, "clods"=> 2.962368e-07, "cloff"=> 1.0444216e-09, "clogs"=> 3.993488e-07, "cloke"=> 1.845898e-07, "clomb"=> 2.401898e-08, "clomp"=> 6.256939999999999e-08, "clone"=> 2.002356e-06, "clonk"=> 1.9947200000000004e-08, "clons"=> 5.188636e-10, "cloop"=> 3.2548819999999996e-09, "cloot"=> 4.775934e-09, "clops"=> 1.282476e-08, "close"=> 0.0001778698, "clote"=> 4.713416e-09, "cloth"=> 1.7212639999999998e-05, "clots"=> 7.460539999999999e-07, "cloud"=> 3.500428e-05, "clour"=> 2.25182e-09, "clous"=> 1.2053275999999999e-08, "clout"=> 1.1050190000000001e-06, "clove"=> 1.55086e-06, "clown"=> 2.5680060000000004e-06, "clows"=> 1.682712e-09, "cloye"=> 2.322424e-10, "cloys"=> 1.0614653999999998e-08, "cloze"=> 1.1045284e-07, "clubs"=> 1.10147e-05, "cluck"=> 2.3373359999999997e-07, "clued"=> 1.795626e-07, "clues"=> 6.255832e-06, "cluey"=> 1.7776219999999997e-09, "clump"=> 1.6517960000000001e-06, "clung"=> 6.388408000000001e-06, "clunk"=> 3.859444e-07, "clype"=> 1.3099200000000002e-09, "cnida"=> 9.7798e-10, "coach"=> 1.9762899999999997e-05, "coact"=> 3.3239999999999997e-09, "coady"=> 1.466922e-07, "coala"=> 1.907572e-09, "coals"=> 2.471862e-06, "coaly"=> 1.8081458000000002e-08, "coapt"=> 8.978744e-09, "coarb"=> 2.4995719999999997e-09, "coast"=> 3.537194e-05, "coate"=> 8.401704000000002e-08, "coati"=> 2.521638e-08, "coats"=> 5.088589999999999e-06, "cobbs"=> 1.280224e-07, "cobby"=> 3.17096e-08, "cobia"=> 2.9406760000000003e-08, "coble"=> 1.0504622e-07, "cobra"=> 1.5531979999999997e-06, "cobza"=> 5.683406e-10, "cocas"=> 2.447334e-09, "cocci"=> 2.0066899999999998e-07, "cocco"=> 4.807438e-08, "cocks"=> 1.4368279999999999e-06, "cocky"=> 1.356496e-06, "cocoa"=> 4.024258e-06, "cocos"=> 2.074524e-07, "codas"=> 7.747906e-08, "codec"=> 2.114102e-07, "coded"=> 5.312679999999999e-06, "coden"=> 6.094332e-09, "coder"=> 4.3735e-07, "codes"=> 1.704958e-05, "codex"=> 2.083258e-06, "codon"=> 7.723948e-07, "coeds"=> 6.136393999999999e-08, "coffs"=> 3.244095999999999e-08, "cogie"=> 2.099062e-09, "cogon"=> 9.709216e-09, "cogue"=> 6.525144e-10, "cohab"=> 8.4844e-09, "cohen"=> 1.175294e-05, "cohoe"=> 3.3244659999999993e-09, "cohog"=> 1.3854222e-10, "cohos"=> 3.271322e-09, "coifs"=> 1.834282e-08, "coign"=> 2.91328e-08, "coils"=> 2.350668e-06, "coins"=> 8.264568e-06, "coirs"=> 1.8075742000000001e-10, "coits"=> 9.899484e-10, "coked"=> 4.049888e-08, "cokes"=> 2.4694140000000006e-07, "colas"=> 1.6980420000000001e-07, "colby"=> 1.373884e-06, "colds"=> 6.95754e-07, "coled"=> 1.989998e-09, "coles"=> 8.574352e-07, "coley"=> 1.97734e-07, "colic"=> 8.699518e-07, "colin"=> 8.727272e-06, "colls"=> 7.559812e-08, "colly"=> 5.7714559999999996e-08, "colog"=> 7.780092e-10, "colon"=> 5.627899999999999e-06, "color"=> 6.10274e-05, "colts"=> 6.175098000000001e-07, "colza"=> 2.442394e-08, "comae"=> 8.663128e-09, "comal"=> 5.216342e-08, "comas"=> 1.661316e-07, "combe"=> 3.9181860000000003e-07, "combi"=> 1.0018473999999999e-07, "combo"=> 9.215681999999999e-07, "combs"=> 1.1397020000000001e-06, "comby"=> 8.517944e-09, "comer"=> 1.0865666e-06, "comes"=> 0.00010834040000000001, "comet"=> 2.266928e-06, "comfy"=> 8.923826e-07, "comic"=> 8.973406e-06, "comix"=> 6.593686e-08, "comma"=> 1.7219239999999998e-06, "commo"=> 4.756644e-08, "comms"=> 4.922741999999999e-07, "commy"=> 6.7236320000000005e-09, "compo"=> 1.462976e-07, "comps"=> 1.654892e-07, "compt"=> 5.825702e-08, "comte"=> 2.230754e-06, "comus"=> 2.20529e-07, "conch"=> 5.179536e-07, "condo"=> 1.960344e-06, "coned"=> 4.09195e-08, "cones"=> 2.3143540000000002e-06, "coney"=> 5.284530000000001e-07, "confs"=> 2.328394e-09, "conga"=> 1.3139000000000001e-07, "conge"=> 1.246914e-08, "congo"=> 5.012492e-06, "conia"=> 6.134484e-09, "conic"=> 3.78734e-07, "conin"=> 1.2092594e-08, "conks"=> 1.2880256e-08, "conky"=> 1.0262006e-08, "conne"=> 2.3489800000000003e-08, "conns"=> 4.351026e-09, "conte"=> 6.253315999999999e-07, "conto"=> 7.82347e-08, "conus"=> 2.677142e-07, "convo"=> 4.40837e-08, "cooch"=> 8.32923e-08, "cooed"=> 7.489226e-07, "cooee"=> 2.706426e-08, "cooer"=> 6.515394e-10, "cooey"=> 1.034028e-08, "coofs"=> 4.360808e-10, "cooks"=> 2.77359e-06, "cooky"=> 6.667033999999999e-08, "cools"=> 9.112065999999999e-07, "cooly"=> 3.875494e-08, "coomb"=> 2.1126179999999998e-08, "cooms"=> 8.917136e-09, "coomy"=> 7.015272e-10, "coons"=> 1.987976e-07, "coops"=> 1.872188e-07, "coopt"=> 4.802022e-08, "coost"=> 3.0515639999999998e-09, "coots"=> 7.9705e-08, "cooze"=> 7.533e-09, "copal"=> 8.131670000000001e-08, "copay"=> 4.484726e-08, "coped"=> 5.747596e-07, "copen"=> 3.09126e-08, "coper"=> 3.2209979999999996e-08, "copes"=> 2.48888e-07, "coppy"=> 4.7454320000000005e-08, "copra"=> 2.309022e-07, "copse"=> 7.202032e-07, "copsy"=> 6.12379e-10, "coqui"=> 2.6411579999999997e-08, "coral"=> 5.540456e-06, "coram"=> 2.773658e-07, "corbe"=> 2.7568848e-09, "corby"=> 1.977388e-07, "cords"=> 3.037204e-06, "cored"=> 4.864459999999999e-07, "corer"=> 4.1767379999999995e-08, "cores"=> 2.7854520000000002e-06, "corey"=> 2.0094500000000003e-06, "corgi"=> 1.402868e-07, "coria"=> 3.709288e-08, "corks"=> 2.708822e-07, "corky"=> 2.850624e-07, "corms"=> 8.057674e-08, "corni"=> 1.9151040000000003e-08, "corno"=> 6.168846e-08, "corns"=> 2.3422319999999998e-07, "cornu"=> 1.5377739999999998e-07, "corny"=> 5.638393999999999e-07, "corps"=> 1.562766e-05, "corse"=> 2.2135160000000001e-07, "corso"=> 5.849296e-07, "cosec"=> 1.5819164e-08, "cosed"=> 7.133574e-10, "coses"=> 8.27306e-09, "coset"=> 1.1723513999999998e-07, "cosey"=> 5.4112439999999996e-08, "cosie"=> 1.3328724e-09, "costa"=> 6.34743e-06, "coste"=> 1.561568e-07, "costs"=> 7.777898e-05, "cotan"=> 3.52435e-09, "coted"=> 8.454602e-09, "cotes"=> 1.0745460000000001e-07, "coths"=> 1.7548684000000002e-10, "cotta"=> 6.605354e-07, "cotts"=> 2.906292e-08, "couch"=> 1.6695120000000004e-05, "coude"=> 4.839636e-08, "cough"=> 7.2007360000000005e-06, "could"=> 0.001276894, "count"=> 4.3113919999999995e-05, "coupe"=> 5.673006e-07, "coups"=> 7.281498e-07, "courb"=> 5.332062e-10, "courd"=> 3.047246e-10, "coure"=> 5.000282e-09, "cours"=> 5.923295999999999e-07, "court"=> 0.0001775598, "couta"=> 3.120678e-09, "couth"=> 2.5965800000000005e-08, "coved"=> 2.694888e-08, "coven"=> 9.591494000000001e-07, "cover"=> 7.158258e-05, "coves"=> 3.274008e-07, "covet"=> 6.243104e-07, "covey"=> 6.394426e-07, "covin"=> 7.298345999999999e-08, "cowal"=> 1.2036219999999998e-08, "cowan"=> 1.201704e-06, "cowed"=> 6.272124e-07, "cower"=> 4.1995439999999997e-07, "cowks"=> 6.450083999999999e-11, "cowls"=> 7.150606e-08, "cowps"=> 1.4047716e-09, "cowry"=> 4.180104e-08, "coxae"=> 5.8261e-08, "coxal"=> 4.0940959999999996e-08, "coxed"=> 7.895058000000001e-09, "coxes"=> 2.103006e-08, "coxib"=> 6.302835999999999e-09, "coyau"=> 3.5455019999999994e-10, "coyed"=> 9.802154e-10, "coyer"=> 2.703262e-08, "coyly"=> 4.2538860000000003e-07, "coypu"=> 1.3772948000000002e-08, "cozed"=> 1.0329208e-10, "cozen"=> 7.198989999999999e-08, "cozes"=> 7.416762e-10, "cozey"=> 2.743864e-10, "cozie"=> 4.3776240000000005e-09, "craal"=> 2.8225720000000005e-10, "crabs"=> 1.575694e-06, "crack"=> 1.6479580000000003e-05, "craft"=> 1.5111960000000001e-05, "crags"=> 6.52827e-07, "craic"=> 5.70321e-08, "craig"=> 8.761942e-06, "crake"=> 1.673616e-07, "crame"=> 1.112687e-08, "cramp"=> 8.174342000000001e-07, "crams"=> 4.275488e-08, "crane"=> 4.765064e-06, "crank"=> 1.5832519999999999e-06, "crans"=> 2.081046e-08, "crape"=> 2.2846079999999999e-07, "craps"=> 1.9639080000000002e-07, "crapy"=> 2.0134746e-09, "crare"=> 1.102722e-09, "crash"=> 1.2384499999999998e-05, "crass"=> 6.338544e-07, "crate"=> 2.101974e-06, "crave"=> 1.7766819999999998e-06, "crawl"=> 5.096014e-06, "craws"=> 1.530282e-08, "crays"=> 8.36804e-09, "craze"=> 8.866305999999999e-07, "crazy"=> 2.7130679999999996e-05, "creak"=> 1.391816e-06, "cream"=> 2.432024e-05, "credo"=> 7.15194e-07, "creds"=> 4.6738100000000003e-08, "creed"=> 4.441414e-06, "creek"=> 1.6057640000000003e-05, "creel"=> 3.4176859999999997e-07, "creep"=> 5.405744e-06, "crees"=> 1.2911011999999998e-07, "creme"=> 1.3320520000000002e-07, "crems"=> 1.5642096e-09, "crena"=> 2.933728e-09, "crepe"=> 4.1800879999999996e-07, "creps"=> 4.315521999999999e-09, "crept"=> 7.827064e-06, "crepy"=> 4.605546e-09, "cress"=> 3.589962e-07, "crest"=> 5.386412e-06, "crewe"=> 6.973146e-07, "crews"=> 4.433588e-06, "crias"=> 9.296413999999999e-09, "cribs"=> 2.613414e-07, "crick"=> 9.897560000000003e-07, "cried"=> 4.94015e-05, "crier"=> 3.484366e-07, "cries"=> 1.0206854e-05, "crime"=> 6.112290000000001e-05, "crimp"=> 3.782386e-07, "crims"=> 2.4088080000000003e-08, "crine"=> 2.025382e-08, "crios"=> 3.741630800000001e-08, "cripe"=> 1.7975920000000003e-08, "crips"=> 1.225992e-07, "crise"=> 1.939768e-07, "crisp"=> 6.304197999999999e-06, "crith"=> 3.873334e-09, "crits"=> 6.044282e-08, "croak"=> 6.58316e-07, "croci"=> 2.854898e-08, "crock"=> 5.569924e-07, "crocs"=> 1.4014500000000002e-07, "croft"=> 1.097434e-06, "crogs"=> 9.279063800000001e-10, "cromb"=> 2.704188e-09, "crome"=> 7.107498e-08, "crone"=> 7.000125999999999e-07, "cronk"=> 7.468994000000001e-08, "crons"=> 2.6807952e-09, "crony"=> 3.1708740000000003e-07, "crook"=> 2.809202e-06, "crool"=> 1.1776757999999999e-08, "croon"=> 1.430712e-07, "crops"=> 1.4681120000000002e-05, "crore"=> 6.942652e-07, "cross"=> 9.810932e-05, "crost"=> 3.469399999999999e-08, "croup"=> 3.788884e-07, "crout"=> 1.4997139999999997e-08, "crowd"=> 4.1639339999999996e-05, "crown"=> 2.422352e-05, "crows"=> 2.159322e-06, "croze"=> 1.770182e-08, "cruck"=> 1.421302e-08, "crude"=> 8.822302e-06, "crudo"=> 3.583808e-08, "cruds"=> 3.311564e-09, "crudy"=> 1.168247e-09, "cruel"=> 1.417536e-05, "crues"=> 1.399017e-08, "cruet"=> 5.124744e-08, "cruft"=> 3.366146e-08, "crumb"=> 1.0291886e-06, "crump"=> 4.2103360000000004e-07, "crunk"=> 2.1264320000000004e-08, "cruor"=> 2.9964300000000006e-08, "crura"=> 1.2229600000000003e-07, "cruse"=> 2.305758e-07, "crush"=> 7.046590000000001e-06, "crust"=> 5.310186e-06, "crusy"=> 1.1662848e-10, "cruve"=> 2.4627564e-10, "crwth"=> 7.331579999999999e-09, "cryer"=> 1.397528e-07, "crypt"=> 1.3496659999999999e-06, "ctene"=> 3.8704219999999997e-10, "cubby"=> 2.8190179999999997e-07, "cubeb"=> 1.2583618000000001e-08, "cubed"=> 6.125722e-07, "cuber"=> 8.055037999999999e-09, "cubes"=> 3.1015819999999996e-06, "cubic"=> 4.434086e-06, "cubit"=> 5.133778e-07, "cuddy"=> 2.58482e-07, "cuffo"=> 1.5917908e-10, "cuffs"=> 2.16635e-06, "cuifs"=> 3.2827102000000004e-10, "cuing"=> 9.761829999999998e-08, "cuish"=> 4.796616e-10, "cuits"=> 1.2447782e-08, "cukes"=> 1.1702824e-08, "culch"=> 6.4822720000000004e-09, "culet"=> 4.153478e-09, "culex"=> 2.1001420000000003e-07, "culls"=> 6.059534000000001e-08, "cully"=> 1.395152e-07, "culms"=> 7.427007999999999e-08, "culpa"=> 3.0388499999999997e-07, "culti"=> 2.494676e-08, "cults"=> 1.7237880000000002e-06, "culty"=> 7.314842e-08, "cumec"=> 6.3035120000000005e-09, "cumin"=> 1.7675519999999998e-06, "cundy"=> 3.6490640000000003e-08, "cunei"=> 4.90283e-09, "cunit"=> 1.3445884000000001e-09, "cunts"=> 1.4880700000000002e-07, "cupel"=> 7.746871999999999e-09, "cupid"=> 1.261276e-06, "cuppa"=> 2.605572e-07, "cuppy"=> 2.1226092e-08, "curat"=> 3.4545119999999995e-08, "curbs"=> 3.246686e-07, "curch"=> 3.208442e-09, "curds"=> 3.1775360000000003e-07, "curdy"=> 1.75655e-08, "cured"=> 4.702230000000001e-06, "curer"=> 2.6584639999999997e-08, "cures"=> 1.7191240000000002e-06, "curet"=> 2.956666e-08, "curfs"=> 6.840088e-09, "curia"=> 6.027956e-07, "curie"=> 9.038102e-07, "curio"=> 2.8456420000000004e-07, "curli"=> 1.924703e-08, "curls"=> 5.328368e-06, "curly"=> 3.9492039999999995e-06, "curns"=> 3.442758e-09, "curny"=> 5.476063999999999e-10, "currs"=> 2.224682e-09, "curry"=> 4.091574e-06, "curse"=> 1.2518220000000001e-05, "cursi"=> 1.254143e-08, "curst"=> 7.329542e-08, "curve"=> 2.9783119999999998e-05, "curvy"=> 7.304259999999999e-07, "cusec"=> 3.4378099999999997e-09, "cushy"=> 2.42917e-07, "cusks"=> 1.1601154e-10, "cusps"=> 3.5036559999999995e-07, "cuspy"=> 1.6835340000000001e-09, "cusso"=> 1.5173013999999997e-09, "cusum"=> 9.728738e-08, "cutch"=> 3.225808e-08, "cuter"=> 1.3859380000000002e-07, "cutes"=> 7.1884279999999995e-09, "cutey"=> 6.0114320000000004e-09, "cutie"=> 2.967058e-07, "cutin"=> 3.348696e-08, "cutis"=> 2.5111400000000003e-07, "cutto"=> 2.88065e-09, "cutty"=> 1.292794e-07, "cutup"=> 1.536898e-08, "cuvee"=> 6.406408e-09, "cuzes"=> 5.766827199999999e-10, "cwtch"=> 4.5677599999999995e-09, "cyano"=> 1.243152e-07, "cyans"=> 2.022722e-09, "cyber"=> 7.075776e-06, "cycad"=> 5.047394e-08, "cycas"=> 4.1539560000000005e-08, "cycle"=> 4.32497e-05, "cyclo"=> 1.641392e-07, "cyder"=> 3.249248e-08, "cylix"=> 1.2532792e-09, "cymae"=> 5.533213999999999e-10, "cymar"=> 1.8687088e-09, "cymas"=> 1.916458e-10, "cymes"=> 4.238982e-08, "cymol"=> 1.0494801999999998e-09, "cynic"=> 6.164864e-07, "cysts"=> 2.6128739999999997e-06, "cytes"=> 2.9578000000000002e-08, "cyton"=> 3.8805699999999996e-09, "czars"=> 9.465967999999999e-08, "daals"=> 1.879326e-09, "dabba"=> 1.892e-08, "daces"=> 1.5509096e-09, "dacha"=> 2.189244e-07, "dacks"=> 1.1779116e-08, "dadah"=> 3.0630579999999997e-09, "dadas"=> 2.204866e-08, "daddy"=> 1.403024e-05, "dados"=> 9.041588e-08, "daffs"=> 5.9681099999999996e-09, "daffy"=> 1.8298859999999998e-07, "dagga"=> 2.9285160000000002e-08, "daggy"=> 1.807368e-08, "dagos"=> 2.1106640000000003e-08, "dahls"=> 4.499588e-09, "daiko"=> 1.9344480000000002e-08, "daily"=> 7.974085999999998e-05, "daine"=> 9.849614e-08, "daint"=> 4.117810000000001e-09, "dairy"=> 7.633946e-06, "daisy"=> 7.70928e-06, "daker"=> 1.877352e-08, "daled"=> 3.4076379999999994e-09, "dales"=> 3.615364e-07, "dalis"=> 2.2770510000000002e-08, "dalle"=> 2.2490520000000003e-07, "dally"=> 2.9381599999999996e-07, "dalts"=> 1.4565782000000003e-10, "daman"=> 1.477014e-07, "damar"=> 3.366318e-08, "dames"=> 7.331166e-07, "damme"=> 2.58955e-07, "damns"=> 6.915532e-08, "damps"=> 1.97216e-07, "dampy"=> 7.46714e-10, "dance"=> 4.21535e-05, "dancy"=> 1.648376e-07, "dandy"=> 1.3049859999999998e-06, "dangs"=> 7.34451e-09, "danio"=> 7.191174e-08, "danks"=> 5.5807739999999996e-08, "danny"=> 9.657584e-06, "dants"=> 3.094814e-08, "daraf"=> 3.0011540000000003e-10, "darbs"=> 4.9195119999999995e-09, "darcy"=> 4.588094e-06, "dared"=> 9.705754e-06, "darer"=> 1.0292848e-08, "dares"=> 1.5396520000000002e-06, "darga"=> 1.7962604e-08, "dargs"=> 1.2613862e-09, "daric"=> 2.1459440000000002e-08, "daris"=> 1.859293e-08, "darks"=> 9.003948e-08, "darky"=> 9.735329999999999e-08, "darns"=> 1.7505680000000002e-08, "darre"=> 5.55707e-09, "darts"=> 1.586166e-06, "darzi"=> 5.2257759999999993e-08, "dashi"=> 1.517594e-07, "dashy"=> 2.812378e-09, "datal"=> 2.955546e-09, "dated"=> 1.3807360000000002e-05, "dater"=> 3.617772e-08, "dates"=> 1.643896e-05, "datos"=> 1.5431519999999997e-07, "datto"=> 6.7627859999999995e-09, "datum"=> 1.1051672e-06, "daube"=> 8.469194e-08, "daubs"=> 6.786152000000001e-08, "dauby"=> 6.346144e-09, "dauds"=> 3.4603306e-10, "dault"=> 4.529333999999999e-09, "daunt"=> 1.6660139999999997e-07, "daurs"=> 1.7904960000000001e-09, "dauts"=> 8.216536e-11, "daven"=> 3.881632e-08, "davit"=> 7.938856e-08, "dawah"=> 3.398078e-08, "dawds"=> 2.8131164e-10, "dawed"=> 1.0064486e-09, "dawen"=> 5.81082e-09, "dawks"=> 4.182524000000001e-09, "dawns"=> 5.35536e-07, "dawts"=> 7.422162200000001e-10, "dayan"=> 4.89805e-07, "daych"=> 1.2232736e-10, "daynt"=> 1.7450834e-10, "dazed"=> 2.96115e-06, "dazer"=> 2.0833478e-09, "dazes"=> 7.948325999999999e-09, "deads"=> 1.938108e-08, "deair"=> 5.69886e-10, "deals"=> 1.2781919999999998e-05, "dealt"=> 1.461438e-05, "deans"=> 6.985106e-07, "deare"=> 1.0902414e-07, "dearn"=> 5.914656e-09, "dears"=> 4.2887939999999995e-07, "deary"=> 1.9634139999999997e-07, "deash"=> 7.754174e-10, "death"=> 0.0002042458, "deave"=> 4.975952e-09, "deaws"=> 6.935628e-11, "deawy"=> 6.641194e-10, "debag"=> 1.0689496e-09, "debar"=> 1.452244e-07, "debby"=> 2.6835639999999997e-07, "debel"=> 4.220682e-09, "debes"=> 7.107974000000001e-08, "debit"=> 2.435736e-06, "debts"=> 8.143204000000001e-06, "debud"=> 1.3688694e-10, "debug"=> 8.173186e-07, "debur"=> 1.8724191999999998e-09, "debus"=> 8.653574000000001e-08, "debut"=> 2.8083719999999997e-06, "debye"=> 4.070268e-07, "decad"=> 2.6253e-08, "decaf"=> 1.9194100000000004e-07, "decal"=> 1.375488e-07, "decan"=> 5.02824e-08, "decay"=> 9.649292e-06, "decko"=> 3.2351499999999995e-09, "decks"=> 2.3403620000000003e-06, "decor"=> 1.3076579999999999e-06, "decos"=> 1.3159996e-09, "decoy"=> 7.090878e-07, "decry"=> 2.7899019999999997e-07, "dedal"=> 2.648802e-09, "deeds"=> 9.982406000000001e-06, "deedy"=> 1.2963672e-08, "deely"=> 3.9387479999999995e-08, "deems"=> 1.267918e-06, "deens"=> 3.329018800000001e-08, "deeps"=> 3.8477520000000004e-07, "deere"=> 3.9088660000000003e-07, "deers"=> 3.99643e-08, "deets"=> 4.636538e-08, "deeve"=> 8.057871999999999e-10, "deevs"=> 2.2069494e-09, "defat"=> 3.602e-09, "defer"=> 1.9045240000000002e-06, "deffo"=> 6.011391999999999e-09, "defis"=> 1.7123e-09, "defog"=> 6.261634e-09, "degas"=> 3.565346e-07, "degum"=> 1.595924e-09, "degus"=> 2.8841992000000002e-08, "deice"=> 5.687026e-09, "deids"=> 4.5321400000000006e-10, "deify"=> 7.291590000000001e-08, "deign"=> 4.4973399999999994e-07, "deils"=> 5.139135999999999e-09, "deism"=> 3.1968380000000003e-07, "deist"=> 1.898526e-07, "deity"=> 5.699682000000001e-06, "deked"=> 4.3581279999999995e-09, "dekes"=> 3.986592e-09, "dekko"=> 1.03983e-08, "delay"=> 2.369736e-05, "deled"=> 3.30081e-09, "deles"=> 1.4403878e-07, "delfs"=> 7.870156e-09, "delft"=> 7.34758e-07, "delis"=> 1.321612e-07, "dells"=> 1.296374e-07, "delly"=> 6.641261999999999e-08, "delos"=> 3.7872579999999996e-07, "delph"=> 1.0457502e-07, "delta"=> 8.250088000000001e-06, "delts"=> 1.516274e-08, "delve"=> 1.439982e-06, "deman"=> 4.1435979999999995e-08, "demes"=> 9.869278e-08, "demic"=> 4.9716879999999996e-08, "demit"=> 1.0111126e-08, "demob"=> 2.855226e-08, "demoi"=> 4.31779e-08, "demon"=> 1.2192186e-05, "demos"=> 1.017976e-06, "dempt"=> 4.800084e-10, "demur"=> 2.6529879999999996e-07, "denar"=> 4.3577899999999996e-08, "denay"=> 3.454086e-09, "dench"=> 1.1199984e-07, "denes"=> 7.585724000000001e-08, "denet"=> 3.6985159999999993e-09, "denim"=> 1.5921079999999998e-06, "denis"=> 3.3572099999999997e-06, "dense"=> 1.241056e-05, "dents"=> 5.527388e-07, "deoxy"=> 2.094254e-07, "depot"=> 3.2323900000000003e-06, "depth"=> 3.550968e-05, "derat"=> 2.4691420000000002e-09, "deray"=> 2.2349380000000002e-08, "derby"=> 2.4250739999999996e-06, "dered"=> 7.099750000000001e-08, "deres"=> 7.335443800000001e-07, "derig"=> 1.2095952e-09, "derma"=> 3.8385980000000004e-08, "derms"=> 4.1019719999999994e-09, "derns"=> 4.238148e-10, "derny"=> 1.5761014000000002e-09, "deros"=> 1.3307430000000003e-08, "derro"=> 6.0833226e-09, "derry"=> 9.381348000000001e-07, "derth"=> 2.97148e-09, "dervs"=> 7.595449999999999e-11, "desex"=> 2.1295964e-09, "deshi"=> 4.8540940000000005e-08, "desis"=> 2.758402e-08, "desks"=> 2.354976e-06, "desse"=> 8.30205e-08, "deter"=> 3.16736e-06, "detox"=> 6.62541e-07, "deuce"=> 9.661942e-07, "devas"=> 3.191312e-07, "devel"=> 2.845714e-07, "devil"=> 2.439764e-05, "devis"=> 5.2937080000000005e-08, "devon"=> 4.14912e-06, "devos"=> 1.8854580000000002e-07, "devot"=> 5.242186e-09, "dewan"=> 2.31341e-07, "dewar"=> 4.406874e-07, "dewax"=> 3.24721e-09, "dewed"=> 2.2887839999999998e-08, "dexes"=> 1.904798e-09, "dexie"=> 2.099402e-08, "dhaba"=> 3.567886e-08, "dhaks"=> 1.619682e-10, "dhals"=> 2.154222e-09, "dhikr"=> 1.2535797999999998e-07, "dhobi"=> 3.52911e-08, "dhole"=> 5.8864520000000004e-08, "dholl"=> 1.421661e-09, "dhols"=> 1.385407e-09, "dhoti"=> 9.574034e-08, "dhows"=> 8.431786000000001e-08, "dhuti"=> 1.8587131999999999e-09, "diact"=> 5.133711999999999e-10, "dials"=> 7.353406e-07, "diane"=> 4.3930620000000006e-06, "diary"=> 1.3410679999999999e-05, "diazo"=> 9.825156000000002e-08, "dibbs"=> 3.249594e-08, "diced"=> 2.551972e-06, "dicer"=> 1.3184419999999998e-07, "dices"=> 9.572238e-08, "dicey"=> 4.801392000000001e-07, "dicht"=> 1.9934042000000002e-08, "dicks"=> 4.976779999999999e-07, "dicky"=> 5.167639999999999e-07, "dicot"=> 5.892216e-08, "dicta"=> 3.9832719999999995e-07, "dicts"=> 1.4845219999999999e-08, "dicty"=> 4.987393999999999e-09, "diddy"=> 9.523004e-08, "didie"=> 1.1475748e-08, "didos"=> 5.444592e-09, "didst"=> 1.784404e-06, "diebs"=> 5.253712e-10, "diels"=> 3.7650219999999995e-07, "diene"=> 2.839042e-07, "diets"=> 4.060212e-06, "diffs"=> 1.2415106e-08, "dight"=> 5.4705359999999995e-08, "digit"=> 4.170674e-06, "dikas"=> 1.4670122000000002e-09, "diked"=> 1.887662e-08, "diker"=> 8.28415e-09, "dikes"=> 3.973404e-07, "dikey"=> 2.6233640000000004e-09, "dildo"=> 6.836574000000001e-07, "dilli"=> 3.254122e-08, "dills"=> 2.350876e-08, "dilly"=> 3.6725700000000004e-07, "dimbo"=> 1.8694099999999996e-09, "dimer"=> 8.981518e-07, "dimes"=> 3.7787619999999997e-07, "dimly"=> 3.2106719999999997e-06, "dimps"=> 3.5069430000000004e-09, "dinar"=> 2.22418e-07, "dined"=> 2.188524e-06, "diner"=> 3.0273540000000004e-06, "dines"=> 2.4815499999999993e-07, "dinge"=> 2.6098319999999996e-07, "dingo"=> 3.2948999999999997e-07, "dings"=> 1.610138e-07, "dingy"=> 1.457148e-06, "dinic"=> 2.8937599999999997e-09, "dinks"=> 4.363992e-08, "dinky"=> 1.8778360000000001e-07, "dinna"=> 5.088776000000001e-07, "dinos"=> 6.176274e-08, "dints"=> 1.8274599999999998e-08, "diode"=> 2.2561960000000004e-06, "diols"=> 1.2882779999999998e-07, "diota"=> 2.2924956e-09, "dippy"=> 7.246004e-08, "dipso"=> 3.401324e-09, "diram"=> 1.1827536000000002e-09, "direr"=> 1.71651e-08, "dirge"=> 3.62374e-07, "dirke"=> 5.569634e-09, "dirks"=> 2.5564759999999997e-07, "dirls"=> 9.741378e-10, "dirts"=> 6.448908e-09, "dirty"=> 1.886164e-05, "disas"=> 1.3812339999999999e-08, "disci"=> 3.019484e-08, "disco"=> 1.247856e-06, "discs"=> 1.9049600000000001e-06, "dishy"=> 3.559724e-08, "disks"=> 2.2380179999999997e-06, "disme"=> 4.0617360000000005e-09, "dital"=> 5.79239e-10, "ditas"=> 4.524168e-09, "ditch"=> 4.824198e-06, "dited"=> 8.605542e-09, "dites"=> 1.619418e-07, "ditsy"=> 1.525062e-08, "ditto"=> 7.96328e-07, "ditts"=> 3.6024559999999996e-10, "ditty"=> 3.0901280000000003e-07, "ditzy"=> 6.093494e-08, "divan"=> 7.68064e-07, "divas"=> 1.6674720000000001e-07, "dived"=> 1.762588e-06, "diver"=> 1.194212e-06, "dives"=> 1.080576e-06, "divis"=> 2.557654e-08, "divna"=> 2.389182e-09, "divos"=> 7.213806e-09, "divot"=> 9.055982e-08, "divvy"=> 8.505764000000001e-08, "diwan"=> 2.272596e-07, "dixie"=> 1.4197279999999999e-06, "dixit"=> 6.861299999999999e-07, "diyas"=> 1.1214926e-08, "dizen"=> 5.7330120000000005e-09, "dizzy"=> 3.7236999999999996e-06, "djinn"=> 3.499946e-07, "djins"=> 3.0406199999999998e-09, "doabs"=> 2.0151879999999997e-09, "doats"=> 9.704569999999998e-09, "dobby"=> 6.839312000000001e-08, "dobes"=> 9.810372000000001e-09, "dobie"=> 1.4577199999999996e-07, "dobla"=> 4.604946000000001e-09, "dobra"=> 4.4588420000000003e-08, "dobro"=> 4.9401280000000003e-08, "docht"=> 5.1485378e-09, "docks"=> 2.944706e-06, "docos"=> 1.5090911999999998e-09, "docus"=> 1.1506352e-09, "doddy"=> 1.822748e-08, "dodge"=> 3.939022e-06, "dodgy"=> 5.121166e-07, "dodos"=> 3.301126e-08, "doeks"=> 8.635438e-10, "doers"=> 6.533132e-07, "doest"=> 3.771172e-07, "doeth"=> 7.496458e-07, "doffs"=> 1.773154e-08, "dogan"=> 1.424678e-07, "doges"=> 5.352428e-08, "dogey"=> 1.2082865999999999e-10, "doggo"=> 3.4547100000000004e-08, "doggy"=> 4.7579739999999995e-07, "dogie"=> 1.1563218e-08, "dogma"=> 2.34354e-06, "dohyo"=> 2.188946e-09, "doilt"=> 5.498756e-11, "doily"=> 9.340702e-08, "doing"=> 0.0001835552, "doits"=> 4.152946e-09, "dojos"=> 1.0493654e-08, "dolce"=> 4.280938e-07, "dolci"=> 6.507547999999999e-08, "doled"=> 2.8401640000000004e-07, "doles"=> 1.0615324000000001e-07, "dolia"=> 2.0773120000000002e-08, "dolls"=> 3.051418e-06, "dolly"=> 3.5635819999999996e-06, "dolma"=> 4.4960220000000004e-08, "dolor"=> 8.826524e-07, "dolos"=> 2.19601e-08, "dolts"=> 4.9658999999999995e-08, "domal"=> 2.112698e-08, "domed"=> 8.914858e-07, "domes"=> 1.21476e-06, "domic"=> 7.1052999999999994e-09, "donah"=> 2.4883659999999997e-09, "donas"=> 5.337334e-08, "donee"=> 3.7863220000000004e-07, "doner"=> 6.37086e-08, "donga"=> 6.054476000000001e-08, "dongs"=> 2.4932659999999996e-08, "donko"=> 3.583682e-09, "donna"=> 5.13688e-06, "donne"=> 1.8829999999999998e-06, "donny"=> 8.01289e-07, "donor"=> 9.929212e-06, "donsy"=> 1.863413e-10, "donut"=> 7.660496e-07, "doobs"=> 2.2080012e-09, "dooce"=> 1.3599549999999998e-08, "doody"=> 1.3705980000000003e-07, "dooks"=> 6.041869999999999e-09, "doole"=> 1.2403742e-08, "dools"=> 3.6429210000000004e-09, "dooly"=> 3.268414e-08, "dooms"=> 1.3517200000000001e-07, "doomy"=> 8.42237e-09, "doona"=> 2.0019142e-07, "doorn"=> 1.244436e-07, "doors"=> 3.714336e-05, "doozy"=> 1.0089e-07, "dopas"=> 8.543216e-10, "doped"=> 2.6198719999999998e-06, "doper"=> 4.523522e-08, "dopes"=> 4.925186e-08, "dopey"=> 2.115008e-07, "dorad"=> 4.5990919999999997e-10, "dorba"=> 1.0497128e-09, "dorbs"=> 1.2355396e-10, "doree"=> 2.2976719999999998e-08, "dores"=> 5.2720859999999994e-08, "doric"=> 4.138804e-07, "doris"=> 2.6946400000000003e-06, "dorks"=> 3.917096e-08, "dorky"=> 1.0789665999999999e-07, "dorms"=> 7.003938e-07, "dormy"=> 1.4210526e-08, "dorps"=> 2.8492280000000003e-09, "dorrs"=> 6.095066e-10, "dorsa"=> 3.54505e-08, "dorse"=> 1.8603256000000003e-08, "dorts"=> 1.583754e-09, "dorty"=> 1.4097834000000001e-09, "dosai"=> 6.0331979999999996e-09, "dosas"=> 3.680232e-08, "dosed"=> 4.3840580000000003e-07, "doseh"=> 3.8095144000000007e-10, "doser"=> 1.2412900000000001e-08, "doses"=> 9.237172e-06, "dosha"=> 1.0973184000000001e-07, "dotal"=> 2.292746e-08, "doted"=> 4.044832e-07, "doter"=> 5.87823e-09, "dotes"=> 1.173605e-07, "dotty"=> 5.427333999999999e-07, "douar"=> 2.03574e-08, "doubt"=> 7.28667e-05, "douce"=> 2.8181640000000004e-07, "doucs"=> 7.040556e-10, "dough"=> 7.653244e-06, "douks"=> 3.415196e-10, "doula"=> 1.4641304000000001e-07, "douma"=> 3.830432e-08, "doums"=> 4.973656e-10, "doups"=> 1.1485215199999998e-09, "doura"=> 1.4092606000000001e-08, "douse"=> 2.8319680000000003e-07, "douts"=> 2.055864e-09, "doved"=> 4.6183400000000003e-10, "doven"=> 9.918964000000001e-09, "dover"=> 3.1533960000000005e-06, "doves"=> 1.356912e-06, "dovie"=> 1.0898004000000001e-07, "dowar"=> 5.373308e-09, "dowds"=> 1.532506e-08, "dowdy"=> 3.067062e-07, "dowed"=> 3.360082e-09, "dowel"=> 2.693116e-07, "dower"=> 6.444496e-07, "dowie"=> 1.273313e-07, "dowle"=> 8.423588000000001e-09, "dowls"=> 6.386946e-10, "dowly"=> 1.74428e-09, "downa"=> 9.24851e-09, "downs"=> 3.7143e-06, "downy"=> 6.436432000000001e-07, "dowps"=> 9.939524e-10, "dowry"=> 1.71223e-06, "dowse"=> 7.554688e-08, "dowts"=> 9.275317999999999e-11, "doxed"=> 4.4817680000000005e-09, "doxes"=> 3.0048080000000003e-09, "doxie"=> 1.3065354e-08, "doyen"=> 1.803958e-07, "doyly"=> 4.1365940000000005e-09, "dozed"=> 1.195276e-06, "dozen"=> 2.2524019999999998e-05, "dozer"=> 1.565074e-07, "dozes"=> 6.835961999999998e-08, "drabs"=> 6.754792e-08, "drack"=> 1.4663332e-08, "draco"=> 4.0801079999999995e-07, "draff"=> 1.2320508e-08, "draft"=> 1.799448e-05, "drags"=> 1.216506e-06, "drail"=> 9.03733e-10, "drain"=> 9.99499e-06, "drake"=> 6.2631219999999996e-06, "drama"=> 1.920562e-05, "drams"=> 1.581254e-07, "drank"=> 1.2279099999999999e-05, "drant"=> 9.169502e-09, "drape"=> 8.488932e-07, "draps"=> 1.821272e-08, "drats"=> 6.104072e-09, "drave"=> 8.868727999999999e-08, "drawl"=> 7.945922e-07, "drawn"=> 4.734544e-05, "draws"=> 1.3582659999999998e-05, "drays"=> 7.091024e-08, "dread"=> 8.42081e-06, "dream"=> 5.3801559999999994e-05, "drear"=> 1.7598859999999998e-07, "dreck"=> 4.063232e-08, "dreed"=> 6.463796e-09, "dreer"=> 2.6933941999999996e-08, "drees"=> 5.3246359999999995e-08, "dregs"=> 7.958444e-07, "dreks"=> 6.592162599999999e-10, "drent"=> 1.9243519999999996e-08, "drere"=> 9.08601e-10, "dress"=> 4.63966e-05, "drest"=> 1.2111804e-07, "dreys"=> 4.700984e-09, "dribs"=> 4.5320040000000004e-08, "drice"=> 1.9349479999999996e-09, "dried"=> 1.7451299999999997e-05, "drier"=> 9.95502e-07, "dries"=> 9.160043999999999e-07, "drift"=> 9.172496e-06, "drill"=> 6.997574e-06, "drily"=> 9.109633999999999e-07, "drink"=> 5.647384e-05, "drips"=> 7.315126000000001e-07, "dript"=> 1.2195050000000001e-09, "drive"=> 6.78319e-05, "droid"=> 3.1710179999999997e-07, "droil"=> 9.630406e-10, "droit"=> 1.777768e-06, "droke"=> 3.5679739999999997e-09, "drole"=> 7.034743999999999e-09, "droll"=> 5.966877999999999e-07, "drome"=> 1.0018140000000001e-07, "drone"=> 4.039548e-06, "drony"=> 1.174439e-09, "droob"=> 4.340417e-10, "droog"=> 4.9294739999999996e-08, "drook"=> 1.6891812e-09, "drool"=> 6.760202000000001e-07, "droop"=> 8.235505999999999e-07, "drops"=> 1.371628e-05, "dropt"=> 1.3521432e-07, "dross"=> 3.9029880000000004e-07, "drouk"=> 2.3501e-10, "drove"=> 3.2637940000000004e-05, "drown"=> 3.3645140000000005e-06, "drows"=> 6.591172e-09, "drubs"=> 2.927526e-09, "drugs"=> 4.381452e-05, "druid"=> 7.15564e-07, "drums"=> 4.304092000000001e-06, "drunk"=> 1.72876e-05, "drupe"=> 5.4550360000000004e-08, "druse"=> 4.2385279999999995e-08, "drusy"=> 7.2620980000000006e-09, "druxy"=> 3.641742e-10, "dryad"=> 2.3986479999999997e-07, "dryas"=> 1.375934e-07, "dryer"=> 1.6804340000000001e-06, "dryly"=> 1.815008e-06, "dsobo"=> 0.0, "dsomo"=> 6.71733e-11, "duads"=> 1.45863706e-09, "duals"=> 7.716018e-08, "duans"=> 7.987086e-10, "duars"=> 8.018322e-09, "dubbo"=> 4.072604e-08, "ducal"=> 5.882496e-07, "ducat"=> 1.490702e-07, "duces"=> 1.52497e-07, "duchy"=> 1.0646068e-06, "ducks"=> 3.1771979999999998e-06, "ducky"=> 2.035816e-07, "ducts"=> 2.095962e-06, "duddy"=> 8.756062e-08, "duded"=> 5.997861999999999e-09, "dudes"=> 6.505156e-07, "duels"=> 4.589815999999999e-07, "duets"=> 2.693246e-07, "duett"=> 1.0344018e-08, "duffs"=> 1.2083079999999999e-08, "dufus"=> 7.956122000000001e-09, "duing"=> 3.7169740000000003e-09, "duits"=> 1.5005514e-07, "dukas"=> 7.52108e-08, "duked"=> 8.322324e-09, "dukes"=> 1.4079180000000002e-06, "dukka"=> 9.615036e-09, "dulce"=> 4.075269999999999e-07, "dules"=> 1.0873606e-09, "dulia"=> 8.614216e-09, "dulls"=> 1.17178e-07, "dully"=> 6.637206000000001e-07, "dulse"=> 7.363840000000001e-08, "dumas"=> 8.657100000000001e-07, "dumbo"=> 1.549256e-07, "dumbs"=> 9.640892e-09, "dumka"=> 4.645944e-08, "dumky"=> 1.1669252e-09, "dummy"=> 3.3313279999999997e-06, "dumps"=> 9.624018e-07, "dumpy"=> 2.073666e-07, "dunam"=> 3.6107379999999996e-08, "dunce"=> 2.341258e-07, "dunch"=> 1.2955384e-08, "dunes"=> 2.274382e-06, "dungs"=> 5.562709999999999e-09, "dungy"=> 4.580426e-08, "dunks"=> 5.4038900000000006e-08, "dunno"=> 1.523066e-06, "dunny"=> 6.518259999999999e-08, "dunsh"=> 1.0190582e-10, "dunts"=> 2.50709e-09, "duomi"=> 3.7041938e-10, "duomo"=> 5.115042000000001e-07, "duped"=> 7.199726e-07, "duper"=> 7.366131999999998e-08, "dupes"=> 2.477166e-07, "duple"=> 1.1826871999999999e-07, "duply"=> 1.4016418e-09, "duppy"=> 2.1509400000000003e-08, "dural"=> 6.809971999999999e-07, "duras"=> 2.104322e-07, "dured"=> 4.375822e-09, "dures"=> 5.802188000000001e-08, "durgy"=> 4.978518e-10, "durns"=> 1.4167672e-09, "duroc"=> 7.349568e-08, "duros"=> 2.4137380000000002e-08, "duroy"=> 6.914582e-08, "durra"=> 2.12661e-08, "durrs"=> 2.4331534e-09, "durry"=> 9.959304000000001e-09, "durst"=> 6.248584e-07, "durum"=> 2.267494e-07, "durzi"=> 1.4830202e-09, "dusks"=> 1.7416579999999998e-08, "dusky"=> 1.563788e-06, "dusts"=> 3.049844e-07, "dusty"=> 6.416158e-06, "dutch"=> 2.47224e-05, "duvet"=> 7.923984e-07, "duxes"=> 1.1768176e-09, "dwaal"=> 1.3277939999999999e-08, "dwale"=> 5.281733999999999e-09, "dwalm"=> 2.049878e-10, "dwams"=> 1.4039045999999999e-09, "dwang"=> 2.566358e-09, "dwarf"=> 3.797518e-06, "dwaum"=> 2.0119392000000002e-10, "dweeb"=> 4.4718059999999996e-08, "dwell"=> 8.545188e-06, "dwelt"=> 3.4283699999999997e-06, "dwile"=> 1.288184e-09, "dwine"=> 2.3761599999999998e-09, "dyads"=> 5.621410000000001e-07, "dyers"=> 1.73308e-07, "dying"=> 2.580236e-05, "dyked"=> 5.90919e-09, "dykes"=> 5.983912e-07, "dykey"=> 3.738264e-09, "dykon"=> 2.8981762e-10, "dynel"=> 2.300062e-08, "dynes"=> 9.020432e-08, "dzhos"=> 0.0, "eager"=> 1.706082e-05, "eagle"=> 8.802929999999999e-06, "eagre"=> 2.6470720000000003e-09, "ealed"=> 3.842957999999999e-09, "eales"=> 6.006436e-08, "eaned"=> 5.011314e-10, "eards"=> 4.584618e-10, "eared"=> 7.357580000000001e-07, "earls"=> 7.909822e-07, "early"=> 0.00025187500000000004, "earns"=> 1.5868499999999999e-06, "earnt"=> 2.4787719999999997e-08, "earst"=> 5.822854e-09, "earth"=> 0.0001247458, "eased"=> 6.95878e-06, "easel"=> 8.660698e-07, "easer"=> 1.0625078e-08, "eases"=> 7.251550000000001e-07, "easle"=> 2.522158e-10, "easts"=> 1.601244e-08, "eaten"=> 1.4005419999999998e-05, "eater"=> 1.289148e-06, "eathe"=> 3.295716e-09, "eaved"=> 1.2401556000000002e-08, "eaves"=> 1.0465984e-06, "ebbed"=> 6.282352000000001e-07, "ebbet"=> 1.1065786e-09, "ebons"=> 1.588425e-10, "ebony"=> 1.6954679999999997e-06, "ebook"=> 1.8598088000000005e-05, "ecads"=> 4.5571521999999996e-10, "eched"=> 1.1861178000000002e-09, "eches"=> 1.3814602000000002e-09, "echos"=> 6.313726000000001e-08, "eclat"=> 5.35042e-08, "ecrus"=> 7.614788e-11, "edema"=> 5.270346e-06, "edged"=> 4.994322e-06, "edger"=> 5.7925520000000005e-08, "edges"=> 1.78401e-05, "edict"=> 1.800728e-06, "edify"=> 1.7852300000000002e-07, "edile"=> 3.205002e-09, "edits"=> 8.320746e-07, "educe"=> 2.794704e-08, "educt"=> 1.0667968e-08, "eejit"=> 6.170984e-08, "eensy"=> 6.316456e-09, "eerie"=> 2.5272179999999998e-06, "eeven"=> 2.320996e-09, "eevns"=> 0.0, "effed"=> 1.3157505999999999e-08, "egads"=> 8.666313999999999e-09, "egers"=> 2.1235540000000002e-09, "egest"=> 1.9445320000000003e-09, "eggar"=> 2.370734e-08, "egged"=> 2.4102339999999997e-07, "egger"=> 2.5680339999999997e-07, "egmas"=> 0.0, "egret"=> 1.83014e-07, "ehing"=> 5.692876000000001e-10, "eider"=> 1.4985299999999999e-07, "eidos"=> 1.421952e-07, "eight"=> 6.938312e-05, "eigne"=> 1.3855634000000001e-08, "eiked"=> 1.6855722000000003e-10, "eikon"=> 7.096131999999999e-08, "eilds"=> 5.751528e-11, "eisel"=> 1.275444e-08, "eject"=> 5.02315e-07, "ejido"=> 1.0870680000000001e-07, "eking"=> 1.1927299999999999e-07, "ekkas"=> 1.8714404e-09, "elain"=> 3.6298640000000004e-08, "eland"=> 1.5937819999999997e-07, "elans"=> 1.6434e-09, "elate"=> 7.154937999999999e-08, "elbow"=> 1.22635e-05, "elchi"=> 3.6129205999999997e-09, "elder"=> 1.4761979999999999e-05, "eldin"=> 7.826971999999999e-08, "elect"=> 5.346916e-06, "elegy"=> 9.703252e-07, "elemi"=> 1.1752438e-08, "elfed"=> 6.062339999999999e-09, "elfin"=> 3.591328e-07, "eliad"=> 4.199168999999999e-09, "elide"=> 1.998412e-07, "elint"=> 3.1957859999999996e-08, "elite"=> 1.890592e-05, "elmen"=> 4.29848e-09, "eloge"=> 2.378502e-08, "elogy"=> 1.3310499999999999e-09, "eloin"=> 2.85440186e-09, "elope"=> 2.621828e-07, "elops"=> 4.351858e-09, "elpee"=> 1.6640564000000001e-10, "elsin"=> 8.91185e-09, "elude"=> 8.688478e-07, "elute"=> 1.1586258e-07, "elvan"=> 1.803858e-08, "elven"=> 4.87929e-07, "elver"=> 2.428004e-08, "elves"=> 2.19346e-06, "emacs"=> 7.1704e-08, "email"=> 1.898534e-05, "embar"=> 1.59483e-08, "embay"=> 2.38187e-09, "embed"=> 1.332618e-06, "ember"=> 1.430758e-06, "embog"=> 6.499293999999999e-11, "embow"=> 2.704116e-10, "embox"=> 2.9417368e-10, "embus"=> 1.0134362000000002e-09, "emcee"=> 1.6682560000000002e-07, "emeer"=> 7.263575999999999e-09, "emend"=> 1.0174408e-07, "emerg"=> 1.4273095999999998e-06, "emery"=> 1.4880259999999999e-06, "emeus"=> 1.4305265999999998e-09, "emics"=> 1.4552779999999999e-08, "emirs"=> 1.707596e-07, "emits"=> 9.79466e-07, "emmas"=> 1.245566e-08, "emmer"=> 1.543384e-07, "emmet"=> 4.370378e-07, "emmew"=> 6.527128000000001e-10, "emmys"=> 5.883138e-08, "emoji"=> 3.6275038000000004e-07, "emong"=> 6.259134e-09, "emote"=> 6.637148e-08, "emove"=> 6.3967440000000004e-09, "empts"=> 4.50217e-08, "empty"=> 5.297824e-05, "emule"=> 2.919536e-09, "emure"=> 1.3443204e-09, "emyde"=> 1.0488498000000001e-10, "emyds"=> 8.573562000000001e-11, "enact"=> 3.367994e-06, "enarm"=> 2.8960212e-10, "enate"=> 8.651341999999999e-09, "ended"=> 4.5543260000000005e-05, "ender"=> 2.9218339999999996e-07, "endew"=> 1.7351618000000003e-10, "endow"=> 6.866108e-07, "endue"=> 1.97991e-08, "enema"=> 5.189676e-07, "enemy"=> 5.0355540000000005e-05, "enews"=> 1.0200578e-08, "enfix"=> 8.365975999999999e-11, "eniac"=> 1.3073004e-07, "enjoy"=> 4.2062760000000004e-05, "enlit"=> 3.764253e-10, "enmew"=> 1.4358526e-10, "ennog"=> 0.0, "ennui"=> 4.865592e-07, "enoki"=> 4.50156e-08, "enols"=> 1.1644378e-08, "enorm"=> 8.008674000000001e-09, "enows"=> 2.2698378e-10, "enrol"=> 3.356808e-07, "ensew"=> 5.559614e-10, "ensky"=> 3.7977540000000004e-10, "ensue"=> 1.2612180000000002e-06, "enter"=> 5.487864e-05, "entia"=> 4.087082e-08, "entry"=> 3.6037080000000006e-05, "enure"=> 1.3197739999999998e-08, "enurn"=> 1.0587386e-10, "envoi"=> 7.256246e-08, "envoy"=> 1.810756e-06, "enzym"=> 4.66684e-08, "eorls"=> 1.3615958e-09, "eosin"=> 3.0259279999999995e-07, "epact"=> 2.578464e-08, "epees"=> 1.3549720000000001e-09, "ephah"=> 3.26593e-07, "ephas"=> 2.4537728e-09, "ephod"=> 5.005105999999999e-07, "ephor"=> 1.4502021999999998e-08, "epics"=> 8.561856e-07, "epoch"=> 3.3653759999999998e-06, "epode"=> 5.179635999999999e-08, "epopt"=> 2.1963502000000002e-09, "epoxy"=> 1.9022239999999998e-06, "epris"=> 2.0348012e-09, "equal"=> 6.838432e-05, "eques"=> 4.0397880000000004e-08, "equid"=> 3.148238e-08, "equip"=> 1.6959360000000003e-06, "erase"=> 3.015244e-06, "erbia"=> 2.6519239999999997e-09, "erect"=> 5.333058e-06, "erevs"=> 9.997508e-10, "ergon"=> 1.878638e-07, "ergos"=> 5.257416000000001e-09, "ergot"=> 2.1862839999999996e-07, "erhus"=> 2.934936e-10, "erica"=> 2.552102e-06, "erick"=> 3.631872e-07, "erics"=> 6.901340000000001e-09, "ering"=> 1.7031799999999998e-07, "erned"=> 8.276858e-09, "ernes"=> 4.906004000000001e-09, "erode"=> 1.111928e-06, "erose"=> 8.044244e-09, "erred"=> 9.697682e-07, "error"=> 4.605476e-05, "erses"=> 4.875540000000001e-09, "eruct"=> 3.888932e-09, "erugo"=> 1.7728246399999997e-09, "erupt"=> 1.1359564e-06, "eruvs"=> 5.044904e-10, "erven"=> 2.051512e-08, "ervil"=> 5.520576e-09, "escar"=> 2.8234040000000004e-09, "escot"=> 1.7761506e-08, "esile"=> 1.9415186e-09, "eskar"=> 5.8890896e-09, "esker"=> 4.330092e-08, "esnes"=> 3.316642e-09, "essay"=> 2.2870199999999997e-05, "esses"=> 1.0910471999999999e-07, "ester"=> 2.42601e-06, "estoc"=> 8.470891999999999e-09, "estop"=> 1.949184e-08, "estro"=> 4.956066000000001e-08, "etage"=> 8.439924e-09, "etape"=> 1.2712579999999998e-08, "etats"=> 1.465426e-07, "etens"=> 1.7482e-10, "ethal"=> 4.309732e-09, "ether"=> 3.686586e-06, "ethic"=> 4.4023499999999994e-06, "ethne"=> 5.941604e-08, "ethos"=> 4.390962e-06, "ethyl"=> 1.891078e-06, "etics"=> 2.4474840000000004e-08, "etnas"=> 1.294831e-09, "ettin"=> 2.0069439999999997e-08, "ettle"=> 8.30533e-09, "etude"=> 2.802742e-07, "etuis"=> 8.005542e-10, "etwee"=> 4.0228440000000004e-10, "etyma"=> 8.897860000000001e-09, "eughs"=> 7.392092e-11, "euked"=> 0.0, "eupad"=> 1.5960026e-10, "euros"=> 1.9973399999999998e-06, "eusol"=> 1.939442e-09, "evade"=> 2.239262e-06, "evens"=> 1.6965640000000002e-07, "event"=> 7.952582e-05, "evert"=> 2.816638e-07, "every"=> 0.0003752554, "evets"=> 2.8032093999999997e-09, "evhoe"=> 0.0, "evict"=> 3.767072e-07, "evils"=> 3.920968e-06, "evite"=> 1.61822e-08, "evohe"=> 1.6823376000000003e-09, "evoke"=> 3.383708e-06, "ewers"=> 1.1713219999999999e-07, "ewest"=> 6.494814e-09, "ewhow"=> 2.6328622000000004e-10, "ewked"=> 0.0, "exact"=> 2.52359e-05, "exalt"=> 8.849832e-07, "exams"=> 4.132254e-06, "excel"=> 6.2910320000000005e-06, "exeat"=> 2.90187e-08, "execs"=> 1.5439280000000001e-07, "exeem"=> 8.709925999999999e-11, "exeme"=> 3.0315239999999997e-10, "exert"=> 5.479794e-06, "exfil"=> 2.2146e-08, "exies"=> 2.3072832000000005e-09, "exile"=> 1.007281e-05, "exine"=> 3.132886e-08, "exing"=> 2.939306e-09, "exist"=> 5.2841659999999996e-05, "exits"=> 2.5721440000000003e-06, "exode"=> 1.7829420000000002e-08, "exome"=> 2.767396e-07, "exons"=> 4.0782500000000004e-07, "expat"=> 2.6863680000000004e-07, "expel"=> 1.5046419999999998e-06, "expos"=> 1.76132e-07, "extol"=> 3.7966260000000006e-07, "extra"=> 3.939072e-05, "exude"=> 3.56163e-07, "exuls"=> 8.796636e-11, "exult"=> 2.694934e-07, "exurb"=> 7.280982e-09, "eyass"=> 2.0139834e-09, "eyers"=> 2.33318e-08, "eying"=> 2.985364e-07, "eyots"=> 2.9562860000000003e-09, "eyras"=> 1.4425422000000001e-09, "eyres"=> 4.243832e-08, "eyrie"=> 2.249322e-07, "eyrir"=> 9.49643e-10, "ezine"=> 2.860084e-08, "fabby"=> 2.187314e-09, "fable"=> 1.9041279999999998e-06, "faced"=> 3.721996e-05, "facer"=> 7.179134e-08, "faces"=> 3.7292219999999996e-05, "facet"=> 2.670472e-06, "facia"=> 2.486018e-08, "facta"=> 1.7460659999999997e-07, "facts"=> 4.393788e-05, "faddy"=> 2.072854e-08, "faded"=> 1.28624e-05, "fader"=> 5.804886e-07, "fades"=> 1.7542780000000003e-06, "fadge"=> 1.9078427999999998e-08, "fados"=> 5.60961e-09, "faena"=> 1.903299e-08, "faery"=> 3.974104000000001e-07, "faffs"=> 7.963796000000001e-10, "faffy"=> 9.499762e-10, "faggy"=> 1.1187297999999999e-08, "fagin"=> 4.085088e-07, "fagot"=> 8.44416e-08, "faiks"=> 5.664878e-10, "fails"=> 1.3954359999999999e-05, "faine"=> 3.881886e-08, "fains"=> 2.719178e-09, "faint"=> 1.6435219999999996e-05, "fairs"=> 1.6391660000000002e-06, "fairy"=> 1.0369338e-05, "faith"=> 8.657792e-05, "faked"=> 9.466348000000001e-07, "faker"=> 9.139432e-08, "fakes"=> 4.0951240000000004e-07, "fakey"=> 6.946328e-09, "fakie"=> 3.969402e-09, "fakir"=> 1.9302160000000002e-07, "falaj"=> 1.9170294e-08, "falls"=> 2.730176e-05, "false"=> 4.605044e-05, "famed"=> 2.126988e-06, "fames"=> 5.973456e-07, "fanal"=> 1.0274982e-08, "fancy"=> 1.808614e-05, "fands"=> 6.089522e-10, "fanes"=> 3.1513100000000004e-08, "fanga"=> 7.942768e-09, "fango"=> 3.035522e-08, "fangs"=> 2.481364e-06, "fanks"=> 2.7449720000000004e-08, "fanny"=> 4.434778e-06, "fanon"=> 1.403062e-06, "fanos"=> 7.695894e-09, "fanum"=> 1.621826e-08, "faqir"=> 4.871306e-08, "farad"=> 7.824971999999999e-08, "farce"=> 1.52324e-06, "farci"=> 1.20128e-08, "farcy"=> 2.109672e-08, "fards"=> 1.6232911999999999e-09, "fared"=> 1.4966779999999998e-06, "farer"=> 6.46183e-08, "fares"=> 1.4643719999999998e-06, "farle"=> 7.112958e-09, "farls"=> 5.7417799999999996e-09, "farms"=> 1.0027946e-05, "faros"=> 7.351317799999999e-08, "farro"=> 1.718128e-07, "farse"=> 2.77529e-09, "farts"=> 2.5195880000000003e-07, "fasci"=> 4.0786100000000004e-08, "fasti"=> 2.2087660000000003e-07, "fasts"=> 3.9050659999999997e-07, "fatal"=> 1.2019460000000001e-05, "fated"=> 1.7143540000000003e-06, "fates"=> 1.7863779999999998e-06, "fatly"=> 1.7056739999999998e-08, "fatso"=> 7.783186e-08, "fatty"=> 9.65521e-06, "fatwa"=> 4.02512e-07, "faugh"=> 6.254330000000001e-08, "fauld"=> 7.545943999999999e-09, "fault"=> 3.070892e-05, "fauna"=> 2.430702e-06, "fauns"=> 1.0796706000000001e-07, "faurd"=> 3.4618459999999995e-10, "fauts"=> 1.8064976e-09, "fauve"=> 2.7519979999999998e-08, "favas"=> 2.0129760000000002e-08, "favel"=> 1.1538734e-08, "faver"=> 1.3815036000000002e-08, "faves"=> 2.96585e-08, "favor"=> 2.7414699999999997e-05, "favus"=> 2.490504e-08, "fawns"=> 2.1584240000000003e-07, "fawny"=> 4.507568e-09, "faxed"=> 2.0947820000000003e-07, "faxes"=> 1.5614e-07, "fayed"=> 7.086594e-08, "fayer"=> 3.3036660000000004e-08, "fayne"=> 5.760498000000001e-08, "fayre"=> 1.48558e-07, "fazed"=> 2.5658759999999997e-07, "fazes"=> 2.7781139999999997e-08, "feals"=> 4.1646880000000004e-10, "feare"=> 4.388074e-07, "fears"=> 1.8094440000000004e-05, "feart"=> 1.3486577999999998e-08, "fease"=> 6.391646e-10, "feast"=> 1.1086804e-05, "feats"=> 1.38638e-06, "feaze"=> 1.2238932e-09, "fecal"=> 2.01878e-06, "feces"=> 1.5608280000000003e-06, "fecht"=> 2.4618639999999998e-08, "fecit"=> 1.9309559999999998e-07, "fecks"=> 2.552832e-09, "fedex"=> 4.6227139999999997e-07, "feebs"=> 1.2615337999999998e-08, "feeds"=> 4.509828e-06, "feels"=> 3.186769999999999e-05, "feens"=> 4.7176987999999996e-09, "feers"=> 1.449544e-09, "feese"=> 2.1947142e-09, "feeze"=> 1.0708866e-09, "fehme"=> 8.459661399999999e-10, "feign"=> 6.713026e-07, "feint"=> 3.8260220000000004e-07, "feist"=> 1.441118e-07, "felch"=> 3.02651e-08, "felid"=> 2.463876e-08, "fella"=> 1.2775999999999998e-06, "fells"=> 2.863098e-07, "felly"=> 5.461262e-08, "felon"=> 6.31012e-07, "felts"=> 8.551612e-08, "felty"=> 4.9726779999999995e-08, "femal"=> 6.491052e-09, "femes"=> 6.745976e-09, "femme"=> 1.536214e-06, "femmy"=> 2.394712e-09, "femur"=> 1.785674e-06, "fence"=> 1.3009980000000001e-05, "fends"=> 4.142527999999999e-08, "fendy"=> 1.1665722000000001e-09, "fenis"=> 2.8424760000000003e-09, "fenks"=> 2.8939100000000003e-10, "fenny"=> 3.876466e-08, "fents"=> 1.8412195999999998e-09, "feods"=> 1.3195624e-10, "feoff"=> 1.6850530000000002e-09, "feral"=> 1.6771820000000001e-06, "ferer"=> 5.634323999999999e-09, "feres"=> 2.9777300000000002e-08, "feria"=> 1.5745540000000002e-07, "ferly"=> 5.660028e-09, "fermi"=> 1.75891e-06, "ferms"=> 2.920748e-09, "ferns"=> 1.487552e-06, "ferny"=> 5.80557e-08, "ferry"=> 5.99528e-06, "fesse"=> 1.2772383999999998e-08, "festa"=> 3.617172e-07, "fests"=> 4.3589999999999995e-08, "festy"=> 7.434842e-09, "fetal"=> 9.021740000000001e-06, "fetas"=> 1.1567704e-09, "fetch"=> 6.9107e-06, "feted"=> 1.829846e-07, "fetes"=> 9.789685999999999e-08, "fetid"=> 4.4007080000000004e-07, "fetor"=> 2.9526040000000002e-08, "fetta"=> 1.0536148e-08, "fetts"=> 6.688289999999999e-10, "fetus"=> 4.259028e-06, "fetwa"=> 2.01288e-09, "feuar"=> 1.912128e-09, "feuds"=> 5.995345999999999e-07, "feued"=> 1.85652e-09, "fever"=> 2.033988e-05, "fewer"=> 1.9745479999999998e-05, "feyed"=> 2.648042e-10, "feyer"=> 1.8503022e-08, "feyly"=> 3.703026e-10, "fezes"=> 7.1063360000000005e-09, "fezzy"=> 1.5107035999999999e-09, "fiars"=> 1.3557396e-09, "fiats"=> 3.71655e-08, "fiber"=> 1.634886e-05, "fibre"=> 5.214624e-06, "fibro"=> 1.1542294000000001e-07, "fices"=> 2.166868e-08, "fiche"=> 6.97949e-08, "fichu"=> 5.517378e-08, "ficin"=> 1.4738460000000002e-08, "ficos"=> 1.3617466e-09, "ficus"=> 2.692412e-07, "fides"=> 5.762167999999999e-07, "fidge"=> 1.6594202e-08, "fidos"=> 3.4345460000000003e-09, "fiefs"=> 2.479878e-07, "field"=> 0.0001579354, "fiend"=> 1.344711e-06, "fient"=> 6.9751300000000005e-09, "fiere"=> 5.1690900000000004e-08, "fiers"=> 4.446576e-08, "fiery"=> 5.6665060000000005e-06, "fiest"=> 5.014842e-09, "fifed"=> 7.462678000000001e-10, "fifer"=> 1.2252746e-07, "fifes"=> 6.940564000000001e-08, "fifis"=> 2.2793959999999998e-09, "fifth"=> 2.6899880000000002e-05, "fifty"=> 3.668228e-05, "figgy"=> 4.199998e-08, "fight"=> 6.883278000000001e-05, "figos"=> 4.160911e-09, "fiked"=> 1.8872762000000003e-10, "fikes"=> 1.991904e-08, "filar"=> 1.1268824e-08, "filch"=> 8.845349999999999e-08, "filed"=> 1.303516e-05, "filer"=> 2.498192e-07, "files"=> 2.4743660000000002e-05, "filet"=> 3.10658e-07, "filii"=> 1.8794640000000003e-07, "filks"=> 2.9689467999999996e-10, "fille"=> 4.568244e-07, "fillo"=> 9.446166e-09, "fills"=> 5.338653999999999e-06, "filly"=> 3.7023020000000004e-07, "filmi"=> 1.9877880000000002e-08, "films"=> 3.0222120000000005e-05, "filmy"=> 3.443648e-07, "filos"=> 1.4759578000000002e-08, "filth"=> 1.94586e-06, "filum"=> 8.118446e-08, "final"=> 9.949322e-05, "finca"=> 1.629112e-07, "finch"=> 1.948174e-06, "finds"=> 2.701874e-05, "fined"=> 1.494996e-06, "finer"=> 3.816880000000001e-06, "fines"=> 3.378372e-06, "finis"=> 2.519442e-07, "finks"=> 4.377622e-08, "finny"=> 8.6844e-08, "finos"=> 1.747628e-08, "fiord"=> 1.1590584e-07, "fiqhs"=> 1.4625526e-10, "fique"=> 2.2598199999999996e-08, "fired"=> 1.963276e-05, "firer"=> 4.4595500000000004e-08, "fires"=> 1.0940880000000001e-05, "firie"=> 4.416956e-09, "firks"=> 9.227306000000001e-10, "firms"=> 3.447606e-05, "firns"=> 1.5119584e-09, "firry"=> 3.781052e-09, "first"=> 0.0008822656, "firth"=> 9.088382e-07, "fiscs"=> 9.873032e-10, "fishy"=> 7.629442e-07, "fisks"=> 3.860826e-09, "fists"=> 7.925752e-06, "fisty"=> 1.1098885999999999e-08, "fitch"=> 9.628944e-07, "fitly"=> 2.5089919999999996e-07, "fitna"=> 6.253787999999999e-08, "fitte"=> 1.3032232000000002e-08, "fitts"=> 1.596666e-07, "fiver"=> 1.800808e-07, "fives"=> 5.36046e-07, "fixed"=> 5.8929960000000004e-05, "fixer"=> 4.915233999999999e-07, "fixes"=> 1.715162e-06, "fixit"=> 4.360046e-08, "fizzy"=> 3.4483540000000005e-07, "fjeld"=> 2.10693e-08, "fjord"=> 5.054474000000001e-07, "flabs"=> 3.579419999999999e-09, "flack"=> 3.3944540000000003e-07, "flaff"=> 7.297970000000001e-10, "flags"=> 5.155563999999999e-06, "flail"=> 5.090238e-07, "flair"=> 1.428384e-06, "flake"=> 9.22369e-07, "flaks"=> 7.334372e-09, "flaky"=> 5.652552e-07, "flame"=> 1.4874119999999998e-05, "flamm"=> 6.537932e-08, "flams"=> 7.500458e-09, "flamy"=> 8.885528000000001e-09, "flane"=> 1.2816362e-08, "flank"=> 4.544754e-06, "flans"=> 2.2141939999999998e-08, "flaps"=> 2.124392e-06, "flare"=> 3.271168e-06, "flary"=> 9.134534e-10, "flash"=> 1.876178e-05, "flask"=> 2.8820099999999995e-06, "flats"=> 3.1621040000000003e-06, "flava"=> 7.484285999999999e-08, "flawn"=> 3.787344e-09, "flaws"=> 3.7293580000000004e-06, "flawy"=> 4.075706e-10, "flaxy"=> 2.25911e-09, "flays"=> 1.689254e-08, "fleam"=> 7.736969999999999e-09, "fleas"=> 9.447039999999999e-07, "fleck"=> 4.634712e-07, "fleek"=> 1.3041086000000002e-08, "fleer"=> 7.41604e-08, "flees"=> 6.353902e-07, "fleet"=> 1.4876940000000001e-05, "flegs"=> 5.13277e-10, "fleme"=> 8.143866000000001e-10, "flesh"=> 3.445346e-05, "fleur"=> 1.1580958e-06, "flews"=> 4.913492e-09, "flexi"=> 9.439174e-08, "flexo"=> 1.7280318000000002e-08, "fleys"=> 1.3039365999999999e-09, "flick"=> 2.8632239999999998e-06, "flics"=> 1.3628e-08, "flied"=> 2.9308459999999998e-08, "flier"=> 4.4948100000000005e-07, "flies"=> 8.275642e-06, "flimp"=> 1.00409518e-09, "flims"=> 6.448218e-09, "fling"=> 2.4367920000000003e-06, "flint"=> 4.193344e-06, "flips"=> 1.07512e-06, "flirs"=> 1.9759856e-09, "flirt"=> 1.5840019999999998e-06, "flisk"=> 8.2943096e-09, "flite"=> 7.343147999999999e-08, "flits"=> 2.112346e-07, "flitt"=> 2.2402953999999996e-09, "float"=> 6.6234940000000005e-06, "flobs"=> 6.4751484e-10, "flock"=> 6.199398e-06, "flocs"=> 8.789866e-08, "floes"=> 1.946998e-07, "flogs"=> 2.04463e-08, "flong"=> 2.089188e-09, "flood"=> 1.558304e-05, "floor"=> 0.00010559453999999999, "flops"=> 1.175732e-06, "flora"=> 6.752814000000001e-06, "flors"=> 1.8330900000000002e-08, "flory"=> 2.4670900000000004e-07, "flosh"=> 6.113072000000001e-10, "floss"=> 7.114194e-07, "flota"=> 4.057794e-08, "flote"=> 1.0474284000000001e-08, "flour"=> 1.3953960000000001e-05, "flout"=> 2.2369659999999997e-07, "flown"=> 4.215452000000001e-06, "flows"=> 2.28966e-05, "flubs"=> 1.2311679999999999e-08, "flued"=> 4.178132e-09, "flues"=> 1.0807529999999998e-07, "fluey"=> 4.599386e-09, "fluff"=> 8.636338e-07, "fluid"=> 3.550168e-05, "fluke"=> 7.616468000000001e-07, "fluky"=> 1.5601260000000002e-08, "flume"=> 3.38336e-07, "flump"=> 1.0163338000000002e-08, "flung"=> 9.43271e-06, "flunk"=> 8.780068e-08, "fluor"=> 1.23792e-07, "flurr"=> 2.8741784e-10, "flush"=> 6.607084e-06, "flute"=> 2.825524e-06, "fluty"=> 2.16561e-08, "fluyt"=> 7.4340639999999995e-09, "flyby"=> 1.2263839999999998e-07, "flyer"=> 1.386016e-06, "flype"=> 1.8455048000000002e-09, "flyte"=> 3.23667e-08, "foals"=> 4.942348000000001e-07, "foams"=> 7.774528e-07, "foamy"=> 4.5993039999999993e-07, "focal"=> 8.787604e-06, "focus"=> 0.00011842900000000001, "foehn"=> 2.402118e-08, "fogey"=> 4.9563540000000005e-08, "foggy"=> 1.3907100000000001e-06, "fogie"=> 2.491784e-09, "fogle"=> 7.241808e-08, "fogou"=> 2.6574800000000002e-09, "fohns"=> 1.7598402000000002e-10, "foids"=> 2.604177e-10, "foils"=> 4.309476e-07, "foins"=> 4.129462e-09, "foist"=> 1.319826e-07, "folds"=> 5.975904000000001e-06, "foley"=> 2.051028e-06, "folia"=> 2.49893e-07, "folic"=> 1.034181e-06, "folie"=> 2.006508e-07, "folio"=> 1.866576e-06, "folks"=> 1.2661280000000002e-05, "folky"=> 1.362178e-08, "folly"=> 5.605128e-06, "fomes"=> 2.37235e-08, "fonda"=> 4.808879999999999e-07, "fonds"=> 5.467064e-07, "fondu"=> 1.8216836e-08, "fones"=> 2.65022e-08, "fonly"=> 8.060702e-10, "fonts"=> 1.0017429999999998e-06, "foods"=> 2.258798e-05, "foody"=> 2.552936e-08, "fools"=> 4.653658e-06, "foots"=> 6.113238e-08, "footy"=> 2.399564e-07, "foram"=> 1.4986696e-07, "foray"=> 8.855525999999999e-07, "forbs"=> 9.690658e-08, "forby"=> 3.208134e-08, "force"=> 0.000144427, "fordo"=> 7.873654e-09, "fords"=> 3.55093e-07, "forel"=> 6.985506e-08, "fores"=> 5.279116e-08, "forex"=> 3.0291940000000006e-07, "forge"=> 4.193282000000001e-06, "forgo"=> 9.424737999999999e-07, "forks"=> 1.988178e-06, "forky"=> 1.2226844e-08, "forme"=> 5.571642e-07, "forms"=> 0.0001076738, "forte"=> 1.2319580000000001e-06, "forth"=> 5.338970000000001e-05, "forts"=> 2.0361300000000003e-06, "forty"=> 3.554176e-05, "forum"=> 1.400108e-05, "forza"=> 2.0618639999999999e-07, "forze"=> 2.5101800000000004e-08, "fossa"=> 1.7571539999999999e-06, "fosse"=> 6.193029999999999e-07, "fouat"=> 6.874542e-11, "fouds"=> 2.2635082e-10, "fouer"=> 7.219284e-10, "fouet"=> 1.559782e-08, "foule"=> 2.7761618e-07, "fouls"=> 1.5139960000000002e-07, "found"=> 0.0003934374, "fount"=> 4.064196e-07, "fours"=> 1.665948e-06, "fouth"=> 5.837626000000001e-09, "fovea"=> 3.09115e-07, "fowls"=> 9.075996e-07, "fowth"=> 3.92599e-10, "foxed"=> 5.909840000000001e-08, "foxes"=> 1.8147020000000002e-06, "foxie"=> 1.9443132000000003e-08, "foyer"=> 2.9088979999999997e-06, "foyle"=> 1.657322e-07, "foyne"=> 2.0103944e-09, "frabs"=> 8.567728e-11, "frack"=> 7.310665999999999e-08, "fract"=> 7.455414000000001e-08, "frags"=> 2.815308e-08, "frail"=> 3.4691499999999996e-06, "fraim"=> 5.784654000000001e-09, "frame"=> 4.214306e-05, "franc"=> 1.1375899999999999e-06, "frank"=> 3.6676000000000006e-05, "frape"=> 8.572288e-09, "fraps"=> 3.900576e-09, "frass"=> 3.5733160000000004e-08, "frate"=> 1.0108427999999999e-07, "frati"=> 4.277324e-08, "frats"=> 1.761748e-08, "fraud"=> 1.205982e-05, "fraus"=> 1.911218e-08, "frays"=> 4.838834e-08, "freak"=> 3.5384579999999998e-06, "freed"=> 7.340258e-06, "freer"=> 1.3607540000000002e-06, "frees"=> 1.0087864e-06, "freet"=> 7.988882e-09, "freit"=> 6.42098e-09, "fremd"=> 2.94951e-08, "frena"=> 1.8090579999999998e-08, "freon"=> 9.53292e-08, "frere"=> 2.8647779999999997e-07, "fresh"=> 5.5479399999999996e-05, "frets"=> 2.6482180000000003e-07, "friar"=> 1.80047e-06, "fribs"=> 4.2213984000000006e-10, "fried"=> 6.601763999999999e-06, "frier"=> 7.162286e-08, "fries"=> 2.519998e-06, "frigs"=> 1.8588585999999999e-09, "frill"=> 2.1731520000000002e-07, "frise"=> 7.025159999999999e-08, "frisk"=> 4.787113999999999e-07, "frist"=> 7.592690000000001e-08, "frith"=> 8.203492e-07, "frits"=> 1.1697182000000002e-07, "fritt"=> 1.4082968000000001e-08, "fritz"=> 3.216402e-06, "frize"=> 9.266216e-09, "frizz"=> 8.600377999999999e-08, "frock"=> 1.587316e-06, "froes"=> 1.274428e-08, "frogs"=> 2.72815e-06, "frond"=> 1.877062e-07, "frons"=> 1.1793229999999999e-07, "front"=> 0.00018323179999999998, "frore"=> 8.262564000000001e-09, "frorn"=> 1.80913e-09, "frory"=> 1.3012006e-10, "frosh"=> 2.485e-07, "frost"=> 7.283642000000001e-06, "froth"=> 8.509482000000001e-07, "frown"=> 7.180517999999999e-06, "frows"=> 2.4882659999999997e-09, "frowy"=> 3.5575076e-10, "froze"=> 6.8737800000000004e-06, "frugs"=> 3.10933e-10, "fruit"=> 3.514986e-05, "frump"=> 5.451268e-08, "frush"=> 1.887114e-08, "frust"=> 5.0355379999999995e-09, "fryer"=> 7.310706000000001e-07, "fubar"=> 3.791282e-08, "fubby"=> 1.9939919999999998e-10, "fubsy"=> 4.436582e-09, "fucks"=> 5.081996000000001e-07, "fucus"=> 8.26487e-08, "fuddy"=> 5.300922e-08, "fudge"=> 1.0766114e-06, "fudgy"=> 4.92014e-08, "fuels"=> 6.297974e-06, "fuero"=> 5.6266719999999994e-08, "fuffs"=> 9.116882000000001e-11, "fuffy"=> 9.960948000000001e-10, "fugal"=> 6.273638e-08, "fuggy"=> 1.332528e-08, "fugie"=> 2.386738e-09, "fugio"=> 3.385234e-09, "fugle"=> 1.9239054000000002e-08, "fugly"=> 1.6364300000000003e-08, "fugue"=> 6.103594e-07, "fugus"=> 4.0088860000000007e-10, "fujis"=> 1.6110899999999999e-09, "fulls"=> 1.653474e-08, "fully"=> 7.158172e-05, "fumed"=> 6.984888e-07, "fumer"=> 1.3881442e-08, "fumes"=> 2.0105919999999996e-06, "fumet"=> 1.2638186e-08, "fundi"=> 5.14016e-08, "funds"=> 3.5363599999999995e-05, "fundy"=> 1.32142e-07, "fungi"=> 5.215697999999999e-06, "fungo"=> 1.56335e-08, "fungs"=> 1.3605906000000001e-09, "funks"=> 2.120368e-08, "funky"=> 7.134438e-07, "funny"=> 2.104836e-05, "fural"=> 5.036914e-10, "furan"=> 2.5371020000000005e-07, "furca"=> 2.101368e-08, "furls"=> 1.3349960000000001e-08, "furol"=> 3.764872e-09, "furor"=> 3.9599699999999997e-07, "furrs"=> 5.487976000000001e-09, "furry"=> 1.533022e-06, "furth"=> 1.1905420000000001e-07, "furze"=> 2.0733579999999996e-07, "furzy"=> 8.37177e-09, "fused"=> 3.4640459999999998e-06, "fusee"=> 5.503620000000001e-08, "fusel"=> 2.646874e-08, "fuses"=> 9.288267999999999e-07, "fusil"=> 3.87522e-08, "fusks"=> 7.644008e-11, "fussy"=> 8.672620000000001e-07, "fusts"=> 3.35296e-10, "fusty"=> 8.901394e-08, "futon"=> 2.875732e-07, "fuzed"=> 7.2894139999999996e-09, "fuzee"=> 3.993492e-09, "fuzes"=> 3.0746139999999994e-08, "fuzil"=> 5.926826e-10, "fuzzy"=> 1.0508288e-05, "fyces"=> 1.295643e-10, "fyked"=> 1.6879952e-10, "fykes"=> 1.7782944e-09, "fyles"=> 2.690656e-08, "fyrds"=> 1.6323406e-09, "fytte"=> 5.003192e-09, "gabba"=> 5.1002119999999994e-08, "gabby"=> 1.7234780000000001e-06, "gable"=> 1.241742e-06, "gaddi"=> 6.864626e-08, "gades"=> 5.129718e-08, "gadge"=> 1.1986207999999999e-08, "gadid"=> 2.6068640000000003e-09, "gadis"=> 1.4043834e-08, "gadje"=> 9.57822e-09, "gadjo"=> 8.056396e-09, "gadso"=> 1.8419420000000002e-09, "gaffe"=> 1.373512e-07, "gaffs"=> 3.5185740000000005e-08, "gaged"=> 3.183228e-08, "gager"=> 8.416812e-08, "gages"=> 1.852304e-07, "gaids"=> 1.688514e-10, "gaily"=> 1.4416278000000001e-06, "gains"=> 1.563918e-05, "gairs"=> 2.497282e-09, "gaita"=> 5.6328020000000006e-08, "gaits"=> 1.4147600000000002e-07, "gaitt"=> 2.0721343999999997e-10, "gajos"=> 8.276224000000001e-09, "galah"=> 2.284774e-08, "galas"=> 1.1330519999999999e-07, "galax"=> 2.689164e-08, "galea"=> 2.075186e-07, "galed"=> 6.471818e-09, "gales"=> 6.736302e-07, "galls"=> 1.9734499999999997e-07, "gally"=> 5.9251339999999995e-08, "galop"=> 4.963795999999999e-08, "galut"=> 2.729732e-08, "galvo"=> 5.084738e-09, "gamas"=> 6.696702e-09, "gamay"=> 5.129692000000001e-08, "gamba"=> 1.5119240000000002e-07, "gambe"=> 1.9453775999999998e-08, "gambo"=> 1.3906460000000002e-08, "gambs"=> 6.5667979999999995e-09, "gamed"=> 6.094646e-08, "gamer"=> 4.4044639999999997e-07, "games"=> 4.074156e-05, "gamey"=> 5.290936e-08, "gamic"=> 8.305692e-09, "gamin"=> 9.461486e-08, "gamma"=> 5.16669e-06, "gamme"=> 1.3416019999999999e-08, "gammy"=> 5.2232999999999995e-08, "gamps"=> 3.513948e-09, "gamut"=> 9.271084e-07, "ganch"=> 4.655601799999999e-09, "gandy"=> 2.2669659999999999e-07, "ganef"=> 2.478124e-09, "ganev"=> 1.4415220000000001e-08, "gangs"=> 4.147218000000001e-06, "ganja"=> 1.357238e-07, "ganof"=> 1.2330348e-10, "gants"=> 2.310548e-08, "gaols"=> 1.0231304000000001e-07, "gaped"=> 1.683092e-06, "gaper"=> 6.967941999999999e-09, "gapes"=> 1.32792e-07, "gapos"=> 2.5822053e-09, "gappy"=> 4.5154420000000003e-08, "garbe"=> 4.026516e-08, "garbo"=> 2.793494e-07, "garbs"=> 3.89478e-08, "garda"=> 5.260382e-07, "gares"=> 9.825004e-09, "garis"=> 4.493003999999999e-08, "garms"=> 1.0533858e-08, "garni"=> 1.0007366e-07, "garre"=> 1.81903e-08, "garth"=> 1.746292e-06, "garum"=> 4.7864759999999995e-08, "gases"=> 8.221452e-06, "gasps"=> 2.010572e-06, "gaspy"=> 7.54476e-09, "gassy"=> 8.476272e-08, "gasts"=> 7.955758000000001e-10, "gatch"=> 2.328956e-08, "gated"=> 1.783314e-06, "gater"=> 2.5845640000000003e-08, "gates"=> 1.783894e-05, "gaths"=> 9.28105e-10, "gator"=> 4.100402e-07, "gauch"=> 2.2113819999999998e-08, "gaucy"=> 3.789276e-10, "gauds"=> 2.7300939999999998e-08, "gaudy"=> 9.526831999999999e-07, "gauge"=> 8.047802e-06, "gauje"=> 1.5263796000000001e-10, "gault"=> 2.955984e-07, "gaums"=> 1.7407706e-10, "gaumy"=> 2.9172279999999996e-10, "gaunt"=> 2.174244e-06, "gaups"=> 6.854726000000001e-11, "gaurs"=> 4.052414e-09, "gauss"=> 1.2547639999999997e-06, "gauze"=> 1.7240499999999998e-06, "gauzy"=> 3.524724e-07, "gavel"=> 4.288256e-07, "gavot"=> 1.220177e-09, "gawcy"=> 7.089778e-11, "gawds"=> 4.680016e-09, "gawks"=> 2.2591140000000002e-08, "gawky"=> 1.547874e-07, "gawps"=> 6.6908499999999995e-09, "gawsy"=> 3.5014346e-10, "gayal"=> 4.5066480000000005e-09, "gayer"=> 1.7487699999999998e-07, "gayly"=> 3.275682e-07, "gazal"=> 1.2878638e-08, "gazar"=> 4.980938e-09, "gazed"=> 1.1426408e-05, "gazer"=> 1.2030898000000003e-07, "gazes"=> 2.39922e-06, "gazon"=> 1.2257696000000002e-08, "gazoo"=> 3.2743139999999997e-09, "geals"=> 2.2398926000000003e-10, "geans"=> 1.5269020000000002e-09, "geare"=> 9.541428e-09, "gears"=> 2.76308e-06, "geats"=> 4.4474500000000006e-08, "gebur"=> 3.3871406e-09, "gecko"=> 3.33516e-07, "gecks"=> 8.736236e-10, "geeks"=> 3.6359800000000005e-07, "geeky"=> 2.1035399999999998e-07, "geeps"=> 3.0297778e-09, "geese"=> 2.016064e-06, "geest"=> 2.8440199999999993e-07, "geist"=> 8.819286000000001e-07, "geits"=> 1.0572064e-10, "gelds"=> 4.195266e-09, "gelee"=> 6.317822e-09, "gelid"=> 3.176762e-08, "gelly"=> 2.492478e-08, "gelts"=> 1.2873702000000001e-09, "gemel"=> 3.3508720000000003e-09, "gemma"=> 2.6050659999999998e-06, "gemmy"=> 1.7935062e-08, "gemot"=> 5.832135999999999e-09, "genal"=> 1.2803022e-08, "genas"=> 3.3129604000000006e-08, "genes"=> 2.273722e-05, "genet"=> 3.024734e-06, "genic"=> 9.627469999999999e-08, "genie"=> 9.95748e-07, "genii"=> 2.4529280000000004e-07, "genip"=> 1.6387555999999999e-09, "genny"=> 1.348632e-07, "genoa"=> 1.602628e-06, "genom"=> 1.747649e-07, "genre"=> 1.388126e-05, "genro"=> 1.6367020000000003e-08, "gents"=> 4.86942e-07, "genty"=> 2.353318e-08, "genua"=> 2.6972180000000003e-08, "genus"=> 5.727114e-06, "geode"=> 5.39803e-08, "geoid"=> 1.0462264000000001e-07, "gerah"=> 5.612254e-09, "gerbe"=> 1.3148902e-08, "geres"=> 5.85522e-09, "gerle"=> 1.2143921999999998e-08, "germs"=> 1.51074e-06, "germy"=> 1.4863820000000002e-08, "gerne"=> 7.263514e-08, "gesse"=> 1.4259504e-08, "gesso"=> 9.304068e-08, "geste"=> 2.094942e-07, "gests"=> 4.589846e-08, "getas"=> 4.612614000000001e-09, "getup"=> 1.766036e-07, "geums"=> 8.960422000000001e-10, "geyan"=> 2.311048e-09, "geyer"=> 3.25977e-07, "ghast"=> 3.4726619999999995e-08, "ghats"=> 2.704022e-07, "ghaut"=> 2.622702e-08, "ghazi"=> 2.451746e-07, "ghees"=> 1.1455262e-09, "ghest"=> 3.3289439999999995e-09, "ghost"=> 1.939496e-05, "ghoul"=> 4.094318e-07, "ghyll"=> 5.244802e-08, "giant"=> 2.22238e-05, "gibed"=> 2.870292e-08, "gibel"=> 1.5539903999999998e-08, "giber"=> 3.541624e-09, "gibes"=> 9.459672e-08, "gibli"=> 2.749412e-09, "gibus"=> 8.582492e-09, "giddy"=> 1.981688e-06, "gifts"=> 1.905384e-05, "gigas"=> 1.308082e-07, "gighe"=> 5.892446400000001e-10, "gigot"=> 2.623008e-08, "gigue"=> 2.9100300000000002e-08, "gilas"=> 5.720378000000001e-09, "gilds"=> 9.870654000000001e-08, "gilet"=> 3.51073e-08, "gills"=> 1.182776e-06, "gilly"=> 4.097468e-07, "gilpy"=> 1.9010524000000002e-10, "gilts"=> 1.1348712e-07, "gimel"=> 3.478402e-08, "gimme"=> 6.149775999999999e-07, "gimps"=> 1.0064068000000002e-08, "gimpy"=> 5.329894e-08, "ginch"=> 3.4831401999999997e-09, "ginge"=> 3.704576e-08, "gings"=> 1.4394235999999999e-09, "ginks"=> 6.692046000000001e-09, "ginny"=> 1.733792e-06, "ginzo"=> 4.148708000000001e-09, "gipon"=> 1.2094404e-09, "gippo"=> 3.140562e-09, "gippy"=> 3.6959539999999995e-09, "gipsy"=> 5.768236e-07, "girds"=> 5.027794e-08, "girls"=> 7.286846e-05, "girly"=> 5.552468e-07, "girns"=> 8.554224000000001e-10, "giron"=> 4.338228e-08, "giros"=> 9.507194e-09, "girrs"=> 2.7170876e-10, "girsh"=> 3.4213e-09, "girth"=> 9.838357999999999e-07, "girts"=> 9.290894000000001e-09, "gismo"=> 1.0342076000000002e-08, "gisms"=> 4.201322e-10, "gists"=> 1.2805854e-07, "gitch"=> 6.3837826e-09, "gites"=> 3.490546e-09, "giust"=> 4.358383999999999e-09, "gived"=> 1.7842319999999998e-08, "given"=> 0.0003021832, "giver"=> 1.913814e-06, "gives"=> 7.256803999999999e-05, "gizmo"=> 2.1576179999999999e-07, "glace"=> 1.7856220000000002e-07, "glade"=> 9.161548e-07, "glads"=> 1.1410314e-08, "glady"=> 2.070676e-08, "glaik"=> 2.507367e-10, "glair"=> 5.668804e-09, "glams"=> 3.1363819999999997e-09, "gland"=> 6.526874e-06, "glans"=> 4.0004419999999996e-07, "glare"=> 7.054416e-06, "glary"=> 8.207008e-09, "glass"=> 7.880316e-05, "glaum"=> 1.0273314e-08, "glaur"=> 2.8143779999999997e-09, "glaze"=> 1.6797099999999998e-06, "glazy"=> 4.697461999999999e-09, "gleam"=> 3.579124e-06, "glean"=> 8.634564e-07, "gleba"=> 2.986728e-08, "glebe"=> 1.7388799999999998e-07, "gleby"=> 5.503226e-10, "glede"=> 4.8759884000000004e-08, "gleds"=> 8.025404e-10, "gleed"=> 2.2844264e-08, "gleek"=> 8.956468e-09, "glees"=> 3.968192e-08, "gleet"=> 8.889732e-09, "gleis"=> 5.613214e-09, "glens"=> 2.687006e-07, "glent"=> 3.261174e-09, "gleys"=> 3.716286e-09, "glial"=> 8.920568e-07, "glias"=> 3.554806e-10, "glibs"=> 6.021978000000001e-10, "glide"=> 2.2780619999999996e-06, "gliff"=> 4.02207e-09, "glift"=> 9.099082600000001e-10, "glike"=> 6.429684e-10, "glime"=> 3.1521939999999997e-09, "glims"=> 6.2628699999999995e-09, "glint"=> 2.028712e-06, "glisk"=> 4.1160840000000006e-09, "glits"=> 8.294699999999999e-10, "glitz"=> 1.668702e-07, "gloam"=> 1.5060844e-08, "gloat"=> 3.8842839999999996e-07, "globe"=> 1.2371960000000002e-05, "globi"=> 1.423108e-08, "globs"=> 1.13236e-07, "globy"=> 1.8510182e-09, "glode"=> 7.552675999999999e-09, "glogg"=> 3.0477640000000003e-09, "gloms"=> 2.579954e-09, "gloom"=> 5.415357999999999e-06, "gloop"=> 4.3916740000000005e-08, "glops"=> 3.83678e-09, "glory"=> 2.791356e-05, "gloss"=> 2.567438e-06, "glost"=> 3.936536e-09, "glout"=> 2.133957e-09, "glove"=> 4.003536e-06, "glows"=> 6.905536e-07, "gloze"=> 7.310826e-09, "glued"=> 2.509434e-06, "gluer"=> 3.337224e-09, "glues"=> 1.9091800000000002e-07, "gluey"=> 6.95521e-08, "glugs"=> 1.67714e-08, "glume"=> 5.99678e-08, "glums"=> 4.11334e-09, "gluon"=> 1.732406e-07, "glute"=> 8.109614e-08, "gluts"=> 5.0405419999999995e-08, "glyph"=> 3.1236520000000004e-07, "gnarl"=> 2.303852e-08, "gnarr"=> 5.8780719999999996e-09, "gnars"=> 1.1960364000000001e-10, "gnash"=> 1.0882859999999998e-07, "gnats"=> 3.2968779999999994e-07, "gnawn"=> 4.827382e-09, "gnaws"=> 1.370376e-07, "gnome"=> 7.110246e-07, "gnows"=> 4.893256400000001e-10, "goads"=> 1.238574e-07, "goafs"=> 1.1383432e-09, "goals"=> 5.249976e-05, "goary"=> 6.059624e-10, "goats"=> 5.128643999999999e-06, "goaty"=> 1.3591560000000002e-08, "goban"=> 9.881452e-09, "gobar"=> 1.4675940000000003e-08, "gobbi"=> 5.076568e-08, "gobbo"=> 6.370532e-08, "gobby"=> 2.61485e-08, "gobis"=> 5.48571e-10, "gobos"=> 1.4332829999999997e-08, "godet"=> 6.328895999999999e-08, "godly"=> 3.020308e-06, "godso"=> 5.276817999999999e-10, "goels"=> 4.850302e-10, "goers"=> 5.354420000000001e-07, "goest"=> 3.60946e-07, "goeth"=> 1.1005759999999999e-06, "goety"=> 9.439756000000001e-10, "gofer"=> 5.674559999999999e-08, "goffs"=> 7.3500260000000005e-09, "gogga"=> 1.8527974e-09, "gogos"=> 2.8129179999999998e-08, "goier"=> 7.182222e-11, "going"=> 0.00042392760000000004, "gojis"=> 1.3994858e-09, "golds"=> 2.4015940000000005e-07, "goldy"=> 8.937589999999999e-08, "golem"=> 5.935673999999999e-07, "goles"=> 1.3212100000000002e-08, "golfs"=> 9.025964e-09, "golly"=> 3.3431600000000004e-07, "golpe"=> 6.981688000000001e-08, "golps"=> 7.408317999999999e-11, "gombo"=> 4.873602e-08, "gomer"=> 2.7418579999999995e-07, "gompa"=> 5.232978e-08, "gonad"=> 2.15553e-07, "gonch"=> 3.3410544e-09, "gonef"=> 3.626888e-10, "goner"=> 2.3839179999999997e-07, "gongs"=> 2.1159739999999998e-07, "gonia"=> 9.271507999999999e-09, "gonif"=> 3.6303160000000003e-09, "gonks"=> 1.7810319999999998e-09, "gonna"=> 1.53436e-05, "gonof"=> 2.9147960000000004e-10, "gonys"=> 5.159478e-09, "gonzo"=> 1.820702e-07, "gooby"=> 4.931732e-08, "goods"=> 5.71637e-05, "goody"=> 8.108012e-07, "gooey"=> 5.298678000000001e-07, "goofs"=> 2.569226e-08, "goofy"=> 8.305791999999999e-07, "googs"=> 1.3321810000000002e-09, "gooks"=> 6.801414e-08, "gooky"=> 8.691224e-10, "goold"=> 1.5359039999999998e-07, "gools"=> 2.2206e-09, "gooly"=> 1.7864481999999998e-09, "goons"=> 5.889746e-07, "goony"=> 5.740756000000001e-09, "goops"=> 5.204808e-09, "goopy"=> 3.772504e-08, "goors"=> 2.523498e-10, "goory"=> 9.998274e-11, "goose"=> 4.836776e-06, "goosy"=> 5.20506e-09, "gopak"=> 2.4632220000000003e-09, "gopik"=> 2.2194498000000002e-10, "goral"=> 2.517298e-08, "goras"=> 6.021111999999999e-09, "gored"=> 1.63659e-07, "gores"=> 8.268009999999999e-08, "gorge"=> 2.676004e-06, "goris"=> 3.886986e-08, "gorms"=> 9.328476e-10, "gormy"=> 4.936012e-10, "gorps"=> 5.815028e-10, "gorse"=> 4.976868000000001e-07, "gorsy"=> 1.9735038e-09, "gosht"=> 1.0753184e-08, "gosse"=> 2.5259459999999997e-07, "gotch"=> 6.704148e-08, "goths"=> 8.063998e-07, "gothy"=> 7.810194e-09, "gotta"=> 6.1036499999999996e-06, "gouch"=> 3.8170439999999995e-09, "gouge"=> 4.133986e-07, "gouks"=> 1.6985399999999997e-10, "goura"=> 4.587893999999999e-09, "gourd"=> 7.938176e-07, "gouts"=> 8.282656e-08, "gouty"=> 2.064378e-07, "gowan"=> 2.7086199999999993e-07, "gowds"=> 3.9063436000000006e-10, "gowfs"=> 0.0, "gowks"=> 2.5270200000000002e-09, "gowls"=> 7.942960000000001e-10, "gowns"=> 2.063578e-06, "goxes"=> 9.293618e-11, "goyim"=> 7.473508e-08, "goyle"=> 2.6499420000000003e-08, "graal"=> 1.1263786e-07, "grabs"=> 3.92244e-06, "grace"=> 5.0461500000000005e-05, "grade"=> 2.926332e-05, "grads"=> 1.3604819999999999e-07, "graff"=> 4.4182399999999997e-07, "graft"=> 5.627051999999999e-06, "grail"=> 1.30267e-06, "grain"=> 2.110918e-05, "graip"=> 2.49046e-09, "grama"=> 9.188834e-08, "grame"=> 2.2272024e-08, "gramp"=> 4.8766340000000004e-08, "grams"=> 4.7064279999999995e-06, "grana"=> 1.4812159999999999e-07, "grand"=> 4.274478e-05, "grans"=> 3.760028e-08, "grant"=> 3.7338499999999995e-05, "grape"=> 3.6120559999999997e-06, "graph"=> 2.16223e-05, "grapy"=> 2.2892180000000003e-09, "grasp"=> 1.615062e-05, "grass"=> 2.928982e-05, "grate"=> 2.0024779999999997e-06, "grave"=> 2.3354919999999997e-05, "gravs"=> 2.9976339999999997e-09, "gravy"=> 1.8152100000000003e-06, "grays"=> 5.689388000000001e-07, "graze"=> 1.342618e-06, "great"=> 0.0003802658, "grebe"=> 1.554484e-07, "grebo"=> 1.3888646000000002e-08, "grece"=> 3.683648e-08, "greed"=> 4.936816e-06, "greek"=> 4.791956e-05, "green"=> 0.0001065114, "grees"=> 1.53529e-08, "greet"=> 6.228932e-06, "grege"=> 6.584193999999999e-09, "grego"=> 8.114892e-08, "grein"=> 4.816332e-08, "grens"=> 3.781202e-08, "grese"=> 1.322238e-08, "greve"=> 2.490426e-07, "grews"=> 2.195226e-09, "greys"=> 4.1725060000000007e-07, "grice"=> 6.547145999999999e-07, "gride"=> 3.6422932e-08, "grids"=> 2.1536300000000004e-06, "grief"=> 2.012722e-05, "griff"=> 1.0393972e-06, "grift"=> 3.4793239999999996e-08, "grigs"=> 1.3293428e-08, "grike"=> 6.659136e-09, "grill"=> 4.278865999999999e-06, "grime"=> 9.928978000000001e-07, "grimy"=> 1.0784599999999999e-06, "grind"=> 2.9671120000000003e-06, "grins"=> 2.00725e-06, "griot"=> 8.611556e-08, "gripe"=> 2.819784e-07, "grips"=> 2.5389539999999997e-06, "gript"=> 4.1645436e-09, "gripy"=> 5.5744964e-10, "grise"=> 7.598294e-08, "grist"=> 4.2300780000000004e-07, "grisy"=> 2.1273296e-09, "grith"=> 1.5315982e-08, "grits"=> 6.594688e-07, "grize"=> 5.87485e-09, "groan"=> 5.503787999999999e-06, "groat"=> 1.913212e-07, "grody"=> 2.1863079999999998e-08, "grogs"=> 7.059962000000001e-09, "groin"=> 2.5885140000000003e-06, "groks"=> 1.3538038e-09, "groma"=> 7.242335999999999e-09, "grone"=> 1.5316100000000002e-08, "groof"=> 1.1122534e-08, "groom"=> 3.891504000000001e-06, "grope"=> 4.96885e-07, "gross"=> 2.069382e-05, "grosz"=> 4.0097839999999995e-07, "grots"=> 1.0203752e-08, "grouf"=> 7.211912e-10, "group"=> 0.0002758726, "grout"=> 5.079604000000001e-07, "grove"=> 7.545198e-06, "grovy"=> 5.68655e-10, "growl"=> 3.902264e-06, "grown"=> 3.785098e-05, "grows"=> 1.1788179999999999e-05, "grrls"=> 5.18069e-09, "grrrl"=> 9.345462e-08, "grubs"=> 2.920258e-07, "grued"=> 3.284957e-10, "gruel"=> 4.168376e-07, "grues"=> 5.222172000000001e-09, "grufe"=> 9.192232e-11, "gruff"=> 1.5774059999999999e-06, "grume"=> 1.7180848000000002e-09, "grump"=> 9.054569999999999e-08, "grund"=> 2.672138e-07, "grunt"=> 2.5851680000000002e-06, "gryce"=> 2.2072566000000002e-07, "gryde"=> 1.7698635799999998e-08, "gryke"=> 3.1634132e-09, "grype"=> 1.7527326e-09, "grypt"=> 1.8887742e-10, "guaco"=> 3.996406e-09, "guana"=> 1.1765082000000001e-08, "guano"=> 3.548692e-07, "guans"=> 5.985288e-09, "guard"=> 4.049010000000001e-05, "guars"=> 5.818550000000001e-10, "guava"=> 3.212854000000001e-07, "gucks"=> 1.3689121999999999e-10, "gucky"=> 5.985016e-10, "gudes"=> 6.575046000000001e-09, "guess"=> 4.99065e-05, "guest"=> 1.835932e-05, "guffs"=> 7.0763e-10, "gugas"=> 1.4237713999999998e-09, "guide"=> 5.445087999999999e-05, "guids"=> 1.1429078e-08, "guild"=> 4.037952000000001e-06, "guile"=> 6.9193e-07, "guilt"=> 2.055584e-05, "guimp"=> 1.426574e-09, "guiro"=> 9.688578000000001e-09, "guise"=> 3.1855660000000004e-06, "gulag"=> 5.701449999999999e-07, "gular"=> 6.073344e-08, "gulas"=> 7.104138e-09, "gulch"=> 5.843672e-07, "gules"=> 8.924986e-08, "gulet"=> 4.317469600000001e-09, "gulfs"=> 2.3269039999999997e-07, "gulfy"=> 1.863322e-09, "gulls"=> 1.13015e-06, "gully"=> 1.280928e-06, "gulph"=> 9.07378e-08, "gulps"=> 6.617774e-07, "gulpy"=> 2.3774779999999997e-09, "gumbo"=> 3.278434e-07, "gumma"=> 3.167764e-08, "gummi"=> 4.48817e-08, "gummy"=> 3.686902e-07, "gumps"=> 8.900946e-09, "gundy"=> 5.258711999999999e-08, "gunge"=> 1.8791240000000003e-08, "gungy"=> 2.049044e-09, "gunks"=> 6.426046e-09, "gunky"=> 2.4479600000000003e-08, "gunny"=> 3.511922e-07, "guppy"=> 3.504236e-07, "guqin"=> 2.6832770000000006e-08, "gurdy"=> 7.315016e-08, "gurge"=> 3.218576e-09, "gurls"=> 1.533676e-08, "gurly"=> 4.900176e-09, "gurns"=> 1.763954e-09, "gurry"=> 2.8512420000000006e-08, "gursh"=> 6.0429882e-10, "gurus"=> 6.909094e-07, "gushy"=> 4.39027e-08, "gusla"=> 1.7022858e-09, "gusle"=> 8.885298e-09, "gusli"=> 5.264739999999999e-09, "gussy"=> 6.410578e-08, "gusto"=> 9.449946e-07, "gusts"=> 9.137219999999999e-07, "gusty"=> 3.0135400000000004e-07, "gutsy"=> 2.07616e-07, "gutta"=> 2.1470279999999998e-07, "gutty"=> 9.026302e-09, "guyed"=> 3.724018e-08, "guyle"=> 2.0231e-09, "guyot"=> 1.30001e-07, "guyse"=> 8.011773999999998e-09, "gwine"=> 3.119456e-07, "gyals"=> 7.180028e-10, "gyans"=> 2.3360184000000003e-10, "gybed"=> 4.431384e-09, "gybes"=> 3.709144e-09, "gyeld"=> 5.570218e-11, "gymps"=> 5.632456e-11, "gynae"=> 1.4761160000000002e-08, "gynie"=> 6.856221999999999e-10, "gynny"=> 4.729923800000001e-10, "gynos"=> 3.1930599999999997e-10, "gyoza"=> 2.965202e-08, "gypos"=> 1.2710918000000002e-09, "gyppo"=> 8.003796000000001e-09, "gyppy"=> 3.816582e-09, "gypsy"=> 2.680138e-06, "gyral"=> 3.9474279999999997e-08, "gyred"=> 2.138996e-09, "gyres"=> 7.786618000000001e-08, "gyron"=> 3.29333e-09, "gyros"=> 1.011132e-07, "gyrus"=> 1.2844119999999998e-06, "gytes"=> 2.443149e-10, "gyved"=> 4.711908e-09, "gyves"=> 2.9628720000000004e-08, "haafs"=> 1.4964832e-10, "haars"=> 9.149893600000002e-09, "habit"=> 2.004108e-05, "hable"=> 3.024988e-08, "habus"=> 5.878268000000001e-09, "hacek"=> 3.0459959999999997e-08, "hacks"=> 5.36054e-07, "hadal"=> 2.3865129999999997e-08, "haded"=> 2.7141744e-09, "hades"=> 1.839394e-06, "hadji"=> 1.515588e-07, "hadst"=> 5.004136e-07, "haems"=> 3.9670399999999996e-09, "haets"=> 1.1470542e-10, "haffs"=> 5.289812e-10, "hafiz"=> 3.538388e-07, "hafts"=> 1.815108e-08, "haggs"=> 8.627468e-09, "hahas"=> 1.79143e-09, "haick"=> 9.494392000000001e-09, "haika"=> 3.090094e-09, "haiks"=> 2.161346e-09, "haiku"=> 5.26337e-07, "hails"=> 3.9916839999999996e-07, "haily"=> 4.687612e-09, "hains"=> 4.4816320000000005e-08, "haint"=> 5.449536e-08, "hairs"=> 4.707412e-06, "hairy"=> 3.1570540000000002e-06, "haith"=> 2.957286e-08, "hajes"=> 1.4719124e-10, "hajis"=> 8.99072e-09, "hajji"=> 1.526636e-07, "hakam"=> 5.247942e-08, "hakas"=> 4.415258e-09, "hakea"=> 1.76194e-08, "hakes"=> 2.784604e-08, "hakim"=> 7.311902e-07, "hakus"=> 1.217511e-10, "halal"=> 8.338378000000001e-07, "haled"=> 8.215234e-08, "haler"=> 9.740128000000001e-09, "hales"=> 4.438878e-07, "halfa"=> 9.780388e-08, "halfs"=> 1.255078e-08, "halid"=> 1.4189300000000002e-08, "hallo"=> 6.178046e-07, "halls"=> 5.58712e-06, "halma"=> 2.3286220000000005e-08, "halms"=> 1.1307918000000001e-09, "halon"=> 5.76292e-08, "halos"=> 3.171446e-07, "halse"=> 6.626684e-08, "halts"=> 5.177890000000001e-07, "halva"=> 4.1745660000000006e-08, "halve"=> 4.0849920000000005e-07, "halwa"=> 3.968632e-08, "hamal"=> 3.944934e-08, "hamba"=> 1.5744719999999996e-08, "hamed"=> 1.695158e-07, "hames"=> 1.0435884000000002e-07, "hammy"=> 6.828116000000001e-08, "hamza"=> 4.083726e-07, "hanap"=> 3.729538e-09, "hance"=> 9.004462000000001e-08, "hanch"=> 1.9908208e-09, "hands"=> 0.00022808300000000002, "handy"=> 5.326727999999999e-06, "hangi"=> 4.3886720000000005e-08, "hangs"=> 3.706846e-06, "hanks"=> 7.03357e-07, "hanky"=> 2.899318e-07, "hansa"=> 2.02926e-07, "hanse"=> 1.1164879999999999e-07, "hants"=> 1.6976059999999999e-07, "haole"=> 8.315614e-08, "haoma"=> 1.791856e-08, "hapax"=> 9.223772e-08, "haply"=> 2.942602e-07, "happi"=> 2.098698e-08, "happy"=> 9.546254000000001e-05, "hapus"=> 7.230638000000002e-09, "haram"=> 1.1164024000000003e-06, "hards"=> 6.344948e-08, "hardy"=> 6.201608e-06, "hared"=> 2.7771520000000005e-08, "harem"=> 1.379282e-06, "hares"=> 6.797446e-07, "harim"=> 1.1463980000000002e-07, "harks"=> 1.7992160000000002e-07, "harls"=> 1.7538106000000002e-09, "harms"=> 2.797296e-06, "harns"=> 4.536163999999999e-09, "haros"=> 1.3443068e-08, "harps"=> 5.320442e-07, "harpy"=> 2.9648819999999996e-07, "harry"=> 2.76614e-05, "harsh"=> 1.359028e-05, "harts"=> 1.1537667999999999e-07, "hashy"=> 1.1963302e-09, "hasks"=> 1.899529e-10, "hasps"=> 1.962758e-08, "hasta"=> 9.61983e-07, "haste"=> 6.2310999999999995e-06, "hasty"=> 3.61321e-06, "hatch"=> 5.031166e-06, "hated"=> 2.0282840000000005e-05, "hater"=> 4.0147599999999996e-07, "hates"=> 3.93257e-06, "hatha"=> 1.750438e-07, "hauds"=> 3.3978819999999997e-09, "haufs"=> 3.954732e-10, "haugh"=> 1.0207144e-07, "hauld"=> 8.023093999999999e-09, "haulm"=> 1.0043328000000001e-08, "hauls"=> 2.78891e-07, "hault"=> 1.1031588e-08, "hauns"=> 5.371294e-09, "haunt"=> 2.939552e-06, "hause"=> 4.3321010000000004e-07, "haute"=> 8.637756e-07, "haven"=> 1.19363e-05, "haver"=> 2.0454580000000003e-07, "haves"=> 4.3478299999999996e-07, "havoc"=> 2.419932e-06, "hawed"=> 7.954794000000001e-08, "hawks"=> 1.6794479999999998e-06, "hawms"=> 4.098922e-11, "hawse"=> 6.400146e-08, "hayed"=> 5.140684e-09, "hayer"=> 2.097172e-08, "hayey"=> 1.494083e-09, "hayle"=> 5.674894000000001e-08, "hazan"=> 1.7355340000000001e-07, "hazed"=> 1.235142e-07, "hazel"=> 4.744942e-06, "hazer"=> 2.8438159999999998e-08, "hazes"=> 4.360384e-08, "heads"=> 3.8082560000000004e-05, "heady"=> 1.6228160000000002e-06, "heald"=> 1.566954e-07, "heals"=> 1.23038e-06, "heame"=> 1.0089176000000001e-09, "heaps"=> 2.304904e-06, "heapy"=> 6.415108e-09, "heard"=> 0.00018583640000000002, "heare"=> 5.495655999999999e-07, "hears"=> 6.329299999999999e-06, "heart"=> 0.000236139, "heast"=> 2.7830942e-08, "heath"=> 5.744456e-06, "heats"=> 1.0600380000000002e-06, "heave"=> 1.779192e-06, "heavy"=> 7.15587e-05, "heben"=> 2.253312e-08, "hebes"=> 1.4134999999999997e-08, "hecht"=> 6.520356e-07, "hecks"=> 5.044118e-09, "heder"=> 3.722136e-08, "hedge"=> 6.761506e-06, "hedgy"=> 4.991548e-09, "heeds"=> 1.296528e-07, "heedy"=> 7.13228e-10, "heels"=> 1.2993459999999999e-05, "heeze"=> 1.909794e-09, "hefte"=> 3.8860460000000006e-08, "hefts"=> 3.077676e-08, "hefty"=> 1.345622e-06, "heids"=> 6.227816e-09, "heigh"=> 1.1219852000000002e-07, "heils"=> 3.076114e-08, "heirs"=> 3.813334e-06, "heist"=> 4.181572e-07, "hejab"=> 1.0129078e-08, "hejra"=> 2.1042062e-09, "heled"=> 2.0143700000000002e-08, "heles"=> 5.58807e-09, "helio"=> 8.517932e-08, "helix"=> 2.1204460000000005e-06, "hello"=> 1.7276659999999997e-05, "hells"=> 6.650410000000001e-07, "helms"=> 7.930957999999999e-07, "helos"=> 5.217408e-08, "helot"=> 4.9566639999999994e-08, "helps"=> 3.445502e-05, "helve"=> 2.6359379999999997e-08, "hemal"=> 1.6513566000000002e-08, "hemes"=> 3.295954e-08, "hemic"=> 2.7283840000000003e-08, "hemin"=> 5.8244439999999997e-08, "hemps"=> 2.9427959999999998e-09, "hempy"=> 1.9390366e-09, "hence"=> 5.586689999999999e-05, "hench"=> 1.0216263999999999e-07, "hends"=> 2.476698e-09, "henge"=> 9.371965999999998e-08, "henna"=> 3.303598e-07, "henny"=> 2.2530560000000002e-07, "henry"=> 5.8244260000000005e-05, "hents"=> 7.239855999999999e-10, "hepar"=> 1.938216e-08, "herbs"=> 7.1412500000000004e-06, "herby"=> 1.0237884e-07, "herds"=> 2.798132e-06, "heres"=> 1.1787100000000001e-07, "herls"=> 1.9395359999999997e-09, "herma"=> 3.991502e-08, "herms"=> 6.288979999999999e-08, "herns"=> 6.94358e-09, "heron"=> 1.5070380000000003e-06, "heros"=> 7.149727999999999e-08, "herry"=> 4.3464520000000005e-08, "herse"=> 3.807953999999999e-08, "hertz"=> 1.122038e-06, "herye"=> 1.470453e-10, "hesps"=> 1.0718315999999999e-10, "hests"=> 4.449732e-09, "hetes"=> 6.946654e-09, "heths"=> 1.9055911e-09, "heuch"=> 4.105932e-09, "heugh"=> 3.263162e-08, "hevea"=> 7.599646e-08, "hewed"=> 2.4539000000000006e-07, "hewer"=> 8.421836e-08, "hewgh"=> 5.94572e-10, "hexad"=> 1.7982954e-08, "hexed"=> 4.1901540000000006e-08, "hexer"=> 1.0926013999999998e-08, "hexes"=> 5.894172e-08, "hexyl"=> 7.347740000000001e-08, "heyed"=> 3.430702e-10, "hiant"=> 5.2515244e-10, "hicks"=> 2.312204e-06, "hided"=> 1.55631e-08, "hider"=> 7.062936e-08, "hides"=> 3.539546e-06, "hiems"=> 1.0538004000000001e-08, "highs"=> 9.859264e-07, "hight"=> 2.3590460000000003e-07, "hijab"=> 6.002996e-07, "hijra"=> 1.786062e-07, "hiked"=> 9.275440000000001e-07, "hiker"=> 4.0190879999999997e-07, "hikes"=> 9.99058e-07, "hikoi"=> 3.744518e-09, "hilar"=> 3.32045e-07, "hilch"=> 5.35419e-10, "hillo"=> 2.0172241999999998e-08, "hills"=> 2.1666820000000002e-05, "hilly"=> 1.065799e-06, "hilts"=> 1.6695920000000002e-07, "hilum"=> 2.6232019999999997e-07, "hilus"=> 4.925770000000001e-08, "himbo"=> 2.202196e-09, "hinau"=> 1.2086958e-09, "hinds"=> 4.5976800000000004e-07, "hinge"=> 2.0653359999999997e-06, "hings"=> 5.3410299999999996e-08, "hinky"=> 4.4233659999999995e-08, "hinny"=> 3.257276e-08, "hints"=> 6.932545999999999e-06, "hiois"=> 0.0, "hiply"=> 7.99905e-10, "hippo"=> 7.912258e-07, "hippy"=> 3.359734e-07, "hired"=> 1.507344e-05, "hiree"=> 3.4729700000000003e-09, "hirer"=> 1.112996e-07, "hires"=> 1.198644e-06, "hissy"=> 9.935759999999999e-08, "hists"=> 1.6191720000000001e-09, "hitch"=> 2.014022e-06, "hithe"=> 7.581368e-09, "hived"=> 5.207308e-08, "hiver"=> 1.0112887999999999e-07, "hives"=> 8.380832e-07, "hizen"=> 1.4973162e-08, "hoaed"=> 3.632354e-11, "hoagy"=> 3.946424e-08, "hoard"=> 1.113796e-06, "hoars"=> 2.90379e-09, "hoary"=> 4.888774000000001e-07, "hoast"=> 7.066262e-09, "hobby"=> 2.749e-06, "hobos"=> 7.910279999999999e-08, "hocks"=> 1.427208e-07, "hocus"=> 1.702988e-07, "hodad"=> 2.3895703999999998e-09, "hodja"=> 3.7107460000000004e-08, "hoers"=> 2.348382e-09, "hogan"=> 1.943764e-06, "hogen"=> 2.272648e-08, "hoggs"=> 2.0117808e-08, "hoghs"=> 6.662118000000001e-11, "hohed"=> 7.282358e-11, "hoick"=> 6.4527959999999994e-09, "hoied"=> 9.393556e-11, "hoiks"=> 5.66824e-10, "hoing"=> 7.544702e-09, "hoise"=> 3.042906e-09, "hoist"=> 1.0724059999999998e-06, "hokas"=> 3.0969594000000002e-09, "hoked"=> 1.900668e-09, "hokes"=> 5.124734e-09, "hokey"=> 1.3672959999999999e-07, "hokis"=> 4.5054224000000003e-10, "hokku"=> 1.817188e-08, "hokum"=> 5.4855560000000004e-08, "holds"=> 3.135566e-05, "holed"=> 8.367300000000001e-07, "holes"=> 1.519922e-05, "holey"=> 1.0874900000000002e-07, "holks"=> 4.5955719999999995e-10, "holla"=> 1.0218112e-07, "hollo"=> 7.497368000000001e-08, "holly"=> 7.85362e-06, "holme"=> 1.9849459999999998e-07, "holms"=> 2.679318e-08, "holon"=> 1.0896612e-07, "holos"=> 5.217248e-08, "holts"=> 4.969834e-08, "homas"=> 1.510868e-07, "homed"=> 1.8471800000000003e-07, "homer"=> 5.7192219999999996e-06, "homes"=> 2.5029220000000002e-05, "homey"=> 4.7153459999999995e-07, "homie"=> 1.1712390000000002e-07, "homme"=> 2.101668e-06, "homos"=> 6.735626e-08, "honan"=> 9.673803999999999e-08, "honda"=> 1.465738e-06, "honds"=> 6.589686000000001e-09, "honed"=> 1.258662e-06, "honer"=> 2.0694539999999997e-08, "hones"=> 9.626364e-08, "honey"=> 1.918002e-05, "hongi"=> 5.04677e-08, "hongs"=> 1.69264e-08, "honks"=> 1.0115007999999999e-07, "honky"=> 2.0664359999999999e-07, "honor"=> 2.9428880000000004e-05, "hooch"=> 2.164408e-07, "hoods"=> 8.180668e-07, "hoody"=> 5.955012e-08, "hooey"=> 3.9370300000000004e-08, "hoofs"=> 1.1575986000000002e-06, "hooka"=> 3.59236e-09, "hooks"=> 3.5517940000000003e-06, "hooky"=> 1.525362e-07, "hooly"=> 1.693656e-08, "hoons"=> 6.404999999999999e-09, "hoops"=> 9.702648e-07, "hoord"=> 4.476173e-09, "hoors"=> 5.562042000000001e-09, "hoosh"=> 1.882376e-08, "hoots"=> 3.0129e-07, "hooty"=> 2.3866309999999996e-08, "hoove"=> 2.2763039999999997e-09, "hopak"=> 2.8628959999999997e-09, "hoped"=> 3.66029e-05, "hoper"=> 2.661388e-08, "hopes"=> 1.879646e-05, "hoppy"=> 1.638436e-07, "horah"=> 3.1277619999999994e-09, "horal"=> 2.6353179999999997e-09, "horas"=> 2.0424860000000005e-07, "horde"=> 1.613594e-06, "horis"=> 1.268206e-08, "horks"=> 8.902958e-10, "horme"=> 4.772369999999999e-09, "horns"=> 5.7378040000000005e-06, "horny"=> 1.478362e-06, "horse"=> 5.850564000000001e-05, "horst"=> 9.721946e-07, "horsy"=> 2.579688e-08, "hosed"=> 1.257348e-07, "hosel"=> 3.2694940000000005e-09, "hosen"=> 4.927689999999999e-08, "hoser"=> 1.1692416e-08, "hoses"=> 7.435080000000001e-07, "hosey"=> 2.4460400000000007e-08, "hosta"=> 4.4628880000000004e-08, "hosts"=> 9.495898e-06, "hotch"=> 3.633179999999999e-08, "hotel"=> 4.812538e-05, "hoten"=> 2.91822e-09, "hotly"=> 1.5541659999999998e-06, "hotty"=> 9.251160000000001e-09, "houff"=> 2.3315800000000003e-09, "houfs"=> 9.979851999999999e-11, "hough"=> 7.323741999999999e-07, "hound"=> 2.4441240000000003e-06, "houri"=> 5.216602e-08, "hours"=> 0.00012832019999999998, "house"=> 0.00032852959999999997, "houts"=> 6.610534e-08, "hovea"=> 1.1602718000000001e-09, "hoved"=> 6.145204e-08, "hovel"=> 5.122864e-07, "hoven"=> 9.204333999999999e-08, "hover"=> 1.773478e-06, "hoves"=> 5.6042059999999996e-09, "howbe"=> 4.71598e-10, "howdy"=> 3.79938e-07, "howes"=> 4.179014e-07, "howff"=> 6.158016e-09, "howfs"=> 1.1600147999999999e-10, "howks"=> 5.035896e-10, "howls"=> 8.954414e-07, "howre"=> 6.721951999999999e-09, "howso"=> 4.164824e-09, "hoxed"=> 1.6520672e-10, "hoxes"=> 5.791328e-10, "hoyas"=> 1.549122e-08, "hoyed"=> 7.05801e-10, "hoyle"=> 4.43837e-07, "hubby"=> 3.794206e-07, "hucks"=> 2.798928e-08, "hudna"=> 1.021223e-08, "hudud"=> 7.723887999999998e-08, "huers"=> 7.640414e-10, "huffs"=> 2.399346e-07, "huffy"=> 1.3046999999999998e-07, "huger"=> 9.388506e-08, "huggy"=> 3.3665960000000006e-08, "huhus"=> 2.0415106e-10, "huias"=> 4.2918182000000005e-10, "hulas"=> 5.898985999999999e-09, "hules"=> 1.2360112e-09, "hulks"=> 2.2108339999999996e-07, "hulky"=> 6.83341e-09, "hullo"=> 6.176702e-07, "hulls"=> 7.07981e-07, "hully"=> 1.6647172e-08, "human"=> 0.0003340218, "humas"=> 3.6805239999999996e-09, "humfs"=> 0.0, "humic"=> 3.591638e-07, "humid"=> 2.4326399999999996e-06, "humor"=> 1.198942e-05, "humph"=> 7.971477999999999e-07, "humps"=> 2.4301579999999997e-07, "humpy"=> 6.746374e-08, "humus"=> 4.385118e-07, "hunch"=> 1.4653e-06, "hunks"=> 2.238816e-07, "hunky"=> 2.73659e-07, "hunts"=> 1.2241379999999998e-06, "hurds"=> 7.789498e-09, "hurls"=> 2.687252e-07, "hurly"=> 1.2851479999999997e-07, "hurra"=> 4.308282e-08, "hurry"=> 1.65317e-05, "hurst"=> 1.580826e-06, "hurts"=> 5.730402e-06, "hushy"=> 8.684398e-10, "husks"=> 5.446356e-07, "husky"=> 2.29546e-06, "husos"=> 5.363769999999999e-10, "hussy"=> 2.947094e-07, "hutch"=> 5.893736e-07, "hutia"=> 5.3786539999999994e-09, "huzza"=> 6.857204e-08, "huzzy"=> 8.501854e-09, "hwyls"=> 0.0, "hydra"=> 7.036112e-07, "hydro"=> 1.7062780000000001e-06, "hyena"=> 5.722218e-07, "hyens"=> 1.7754598e-10, "hygge"=> 9.122939999999998e-08, "hying"=> 2.668504e-09, "hykes"=> 4.799178e-09, "hylas"=> 1.554732e-07, "hyleg"=> 1.2889567999999999e-09, "hyles"=> 7.756582e-09, "hylic"=> 6.982727999999999e-09, "hymen"=> 3.772038e-07, "hymns"=> 3.174298e-06, "hynde"=> 4.6201920000000007e-08, "hyoid"=> 3.3737040000000003e-07, "hyped"=> 2.774522e-07, "hyper"=> 3.822254e-06, "hypes"=> 1.916496e-08, "hypha"=> 5.0730139999999997e-08, "hyphy"=> 2.9130959999999998e-09, "hypos"=> 1.6556e-08, "hyrax"=> 4.525246e-08, "hyson"=> 4.550538e-08, "hythe"=> 1.0111660000000001e-07, "iambi"=> 1.466764e-08, "iambs"=> 3.3631799999999995e-08, "ibrik"=> 3.2712659999999996e-09, "icers"=> 1.968944e-08, "iched"=> 1.6141796e-10, "iches"=> 3.4662499999999997e-09, "ichor"=> 1.2916139999999997e-07, "icier"=> 2.460886e-08, "icily"=> 2.9642240000000003e-07, "icing"=> 1.9870840000000004e-06, "icker"=> 5.9385e-09, "ickle"=> 2.330602e-08, "icons"=> 3.3117339999999997e-06, "ictal"=> 2.3064239999999998e-07, "ictic"=> 2.644711e-09, "ictus"=> 6.449062e-08, "idant"=> 1.656564e-09, "ideal"=> 3.9110320000000005e-05, "ideas"=> 9.204338000000001e-05, "idees"=> 4.2254079999999995e-08, "ident"=> 1.2146079999999997e-07, "idiom"=> 2.112372e-06, "idiot"=> 7.863608e-06, "idled"=> 2.55394e-07, "idler"=> 3.5833499999999993e-07, "idles"=> 4.0394179999999995e-08, "idola"=> 1.8095078000000003e-08, "idols"=> 3.407828e-06, "idyll"=> 4.894658000000001e-07, "idyls"=> 2.049996e-08, "iftar"=> 4.326860000000001e-08, "igapo"=> 1.0791862000000001e-09, "igged"=> 5.06064e-10, "igloo"=> 2.423026e-07, "iglus"=> 6.308062e-10, "ihram"=> 2.5445040000000003e-08, "ikans"=> 6.397448e-11, "ikats"=> 2.459314e-09, "ikons"=> 5.54805e-08, "ileac"=> 3.88115e-09, "ileal"=> 3.645482e-07, "ileum"=> 5.973223999999999e-07, "ileus"=> 4.273161999999999e-07, "iliac"=> 1.722098e-06, "iliad"=> 1.5662279999999999e-06, "ilial"=> 9.445326e-09, "ilium"=> 3.43016e-07, "iller"=> 5.5136679999999997e-08, "illth"=> 3.6692120000000006e-09, "image"=> 0.0001154822, "imago"=> 7.133364e-07, "imams"=> 5.291248e-07, "imari"=> 3.815672e-08, "imaum"=> 1.0275814e-08, "imbar"=> 2.39734146e-08, "imbed"=> 3.1245959999999995e-08, "imbue"=> 3.687282e-07, "imide"=> 1.1416976e-07, "imido"=> 1.9794500000000002e-08, "imids"=> 1.1086236e-08, "imine"=> 2.282648e-07, "imino"=> 7.319906e-08, "immew"=> 9.371466e-11, "immit"=> 3.81936e-10, "immix"=> 1.0287578000000002e-09, "imped"=> 7.930308000000001e-09, "impel"=> 2.76137e-07, "impis"=> 1.6466439999999998e-08, "imply"=> 1.12441e-05, "impot"=> 2.23948e-08, "impro"=> 3.2562839999999994e-08, "imshi"=> 6.661132e-09, "imshy"=> 1.035413e-10, "inane"=> 4.6282300000000004e-07, "inapt"=> 6.522259999999999e-08, "inarm"=> 9.559639999999999e-09, "inbox"=> 7.526403999999999e-07, "inbye"=> 3.652974e-09, "incel"=> 1.1639752e-08, "incle"=> 5.894182e-09, "incog"=> 2.36957e-08, "incur"=> 2.1846899999999997e-06, "incus"=> 1.3688900000000003e-07, "incut"=> 1.7525199999999997e-09, "indew"=> 1.92228e-10, "index"=> 5.584756e-05, "india"=> 7.946974e-05, "indie"=> 9.540672e-07, "indol"=> 3.618862e-08, "indow"=> 8.25502e-09, "indri"=> 2.222364e-08, "indue"=> 7.1122359999999986e-09, "inept"=> 8.402978000000001e-07, "inerm"=> 5.470063e-10, "inert"=> 2.840616e-06, "infer"=> 4.191756000000001e-06, "infix"=> 9.38205e-08, "infos"=> 1.756448e-08, "infra"=> 1.5583039999999999e-06, "ingan"=> 1.0547873999999998e-07, "ingle"=> 2.204016e-07, "ingot"=> 3.663878e-07, "inion"=> 3.19922e-08, "inked"=> 5.234198e-07, "inker"=> 3.5537619999999995e-08, "inkle"=> 3.63294e-08, "inlay"=> 2.6018760000000003e-07, "inlet"=> 3.896728000000001e-06, "inned"=> 2.16789e-09, "inner"=> 4.2333179999999995e-05, "innit"=> 1.0935532000000001e-07, "inorb"=> 5.50991e-11, "input"=> 4.182678e-05, "inrun"=> 1.0977198e-09, "inset"=> 1.4864299999999998e-06, "inspo"=> 4.7301439999999996e-09, "intel"=> 2.69939e-06, "inter"=> 1.81771e-05, "intil"=> 2.5036959999999995e-08, "intis"=> 2.3872260000000002e-09, "intra"=> 7.027692e-06, "intro"=> 1.094709e-06, "inula"=> 2.097562e-08, "inure"=> 8.030162e-08, "inurn"=> 7.529286000000001e-10, "inust"=> 2.3363718e-09, "invar"=> 3.8283720000000004e-08, "inwit"=> 1.0765764e-08, "iodic"=> 6.438565999999999e-09, "iodid"=> 2.0564558e-09, "iodin"=> 4.1668486e-09, "ionic"=> 5.003394e-06, "iotas"=> 5.365528e-09, "ippon"=> 1.0332106e-08, "irade"=> 9.554812000000001e-09, "irate"=> 7.011632e-07, "irids"=> 7.042619999999999e-09, "iring"=> 1.3225019999999999e-08, "irked"=> 4.858476000000001e-07, "iroko"=> 2.6859200000000007e-08, "irone"=> 3.7125160000000003e-09, "irons"=> 1.48297e-06, "irony"=> 7.745302e-06, "isbas"=> 1.941889e-09, "ishes"=> 1.2309399999999998e-08, "isled"=> 3.5177740000000004e-09, "isles"=> 2.6958739999999996e-06, "islet"=> 1.0200587999999998e-06, "isnae"=> 2.117958e-08, "issei"=> 1.2085296000000002e-07, "issue"=> 0.00010297513999999999, "istle"=> 1.601076e-09, "itchy"=> 8.960059999999999e-07, "items"=> 4.707462e-05, "ither"=> 9.245656000000001e-08, "ivied"=> 4.606534e-08, "ivies"=> 3.648434e-08, "ivory"=> 5.1215800000000005e-06, "ixias"=> 1.6706646e-09, "ixnay"=> 5.180628e-09, "ixora"=> 1.6081318000000002e-08, "ixtle"=> 2.1822739999999998e-09, "izard"=> 1.741504e-07, "izars"=> 2.9533624e-10, "izzat"=> 6.487673999999999e-08, "jaaps"=> 2.6030292e-10, "jabot"=> 2.7501560000000002e-08, "jacal"=> 2.7929739999999996e-08, "jacks"=> 1.0875316e-06, "jacky"=> 4.3181820000000005e-07, "jaded"=> 7.600566e-07, "jades"=> 7.930836e-08, "jafas"=> 5.674103999999999e-10, "jaffa"=> 6.458161999999999e-07, "jagas"=> 3.2768299999999997e-09, "jager"=> 3.2465779999999996e-07, "jaggs"=> 1.6306714000000003e-08, "jaggy"=> 1.0659362000000001e-08, "jagir"=> 3.196444e-08, "jagra"=> 7.765561999999999e-10, "jails"=> 1.14236e-06, "jaker"=> 4.2402680000000005e-09, "jakes"=> 2.5786979999999993e-07, "jakey"=> 9.468648e-08, "jalap"=> 1.8860420000000003e-08, "jalop"=> 6.086574e-10, "jambe"=> 3.1326199999999996e-08, "jambo"=> 3.948234e-08, "jambs"=> 1.364796e-07, "jambu"=> 2.387776e-08, "james"=> 0.00010163664, "jammy"=> 7.40644e-08, "jamon"=> 2.2854767999999997e-08, "janes"=> 2.3998599999999996e-07, "janns"=> 1.8928056e-09, "janny"=> 6.412444000000001e-08, "janty"=> 1.2211094e-09, "japan"=> 4.5732140000000005e-05, "japed"=> 8.4163e-09, "japer"=> 2.613198e-09, "japes"=> 2.87212e-08, "jarks"=> 1.2418302399999999e-09, "jarls"=> 3.3150539999999996e-08, "jarps"=> 1.383001e-10, "jarta"=> 6.041090000000001e-10, "jarul"=> 1.0005866000000001e-09, "jasey"=> 9.230394e-09, "jaspe"=> 7.308836000000001e-09, "jasps"=> 8.356574e-11, "jatos"=> 7.956272e-09, "jauks"=> 1.683147e-10, "jaunt"=> 3.8135720000000003e-07, "jaups"=> 5.085150000000001e-10, "javas"=> 4.49712e-09, "javel"=> 2.3173760000000002e-08, "jawan"=> 2.84082e-08, "jawed"=> 4.740058e-07, "jaxie"=> 9.895356000000001e-09, "jazzy"=> 3.048434e-07, "jeans"=> 1.39886e-05, "jeats"=> 1.0622632000000003e-10, "jebel"=> 2.55473e-07, "jedis"=> 9.757777999999999e-09, "jeels"=> 5.938074e-10, "jeely"=> 3.712954e-09, "jeeps"=> 4.083768e-07, "jeers"=> 3.278082e-07, "jeeze"=> 2.353084e-08, "jefes"=> 3.7490240000000004e-08, "jeffs"=> 9.667424e-08, "jehad"=> 2.8024400000000002e-08, "jehus"=> 8.3694e-09, "jelab"=> 2.2942422e-10, "jello"=> 1.0277396e-07, "jells"=> 6.548992e-09, "jelly"=> 3.2083279999999998e-06, "jembe"=> 7.581094e-09, "jemmy"=> 3.236226e-07, "jenny"=> 9.861106e-06, "jeons"=> 3.429336e-10, "jerid"=> 6.049173999999999e-09, "jerks"=> 1.233292e-06, "jerky"=> 1.21832e-06, "jerry"=> 9.628191999999998e-06, "jesse"=> 8.009524e-06, "jests"=> 4.1521819999999996e-07, "jesus"=> 0.0001331776, "jetes"=> 1.1981002e-09, "jeton"=> 1.706188e-08, "jetty"=> 1.0529274e-06, "jeune"=> 5.335456e-07, "jewed"=> 2.9250720000000003e-09, "jewel"=> 4.3249619999999995e-06, "jewie"=> 2.3750400000000004e-10, "jhala"=> 3.405036e-08, "jiaos"=> 4.450146e-10, "jibba"=> 3.583866e-09, "jibbs"=> 1.8390177999999998e-09, "jibed"=> 6.395052e-08, "jiber"=> 3.430056e-10, "jibes"=> 1.66581e-07, "jiffs"=> 1.5395394e-09, "jiffy"=> 2.785536e-07, "jiggy"=> 2.271168e-08, "jigot"=> 5.620564999999999e-10, "jihad"=> 2.55635e-06, "jills"=> 2.005636e-08, "jilts"=> 1.1211396e-08, "jimmy"=> 1.222882e-05, "jimpy"=> 1.8892266e-09, "jingo"=> 1.1202384e-07, "jinks"=> 2.2329119999999997e-07, "jinne"=> 3.735498e-09, "jinni"=> 9.467118e-08, "jinns"=> 6.416708000000001e-08, "jirds"=> 4.057672e-09, "jirga"=> 7.842632e-08, "jirre"=> 2.0102238e-09, "jisms"=> 1.3077456e-10, "jived"=> 1.412602e-08, "jiver"=> 1.2188504000000001e-09, "jives"=> 1.0329972000000002e-08, "jivey"=> 1.2756726e-09, "jnana"=> 1.0945146e-07, "jobed"=> 9.085549100000001e-09, "jobes"=> 6.2988e-08, "jocko"=> 9.7801e-08, "jocks"=> 2.8174899999999996e-07, "jocky"=> 1.2075350000000001e-08, "jocos"=> 1.2797108e-09, "jodel"=> 6.679530000000002e-09, "joeys"=> 2.325632e-08, "johns"=> 5.55612e-06, "joins"=> 3.7186539999999997e-06, "joint"=> 4.756398e-05, "joist"=> 2.1949479999999996e-07, "joked"=> 3.028576e-06, "joker"=> 9.530860000000001e-07, "jokes"=> 6.402158e-06, "jokey"=> 1.0964320000000001e-07, "jokol"=> 8.400105800000002e-10, "joled"=> 7.210608000000001e-11, "joles"=> 4.537408e-09, "jolls"=> 1.949992e-08, "jolly"=> 3.956742e-06, "jolts"=> 4.132432e-07, "jolty"=> 5.650632000000001e-09, "jomon"=> 7.196646e-08, "jomos"=> 1.9009360000000002e-10, "jones"=> 3.189728e-05, "jongs"=> 3.341436e-09, "jonty"=> 1.4341276e-07, "jooks"=> 1.679818e-09, "joram"=> 4.888376e-07, "jorum"=> 1.828122e-08, "jotas"=> 3.4940980000000004e-09, "jotty"=> 1.7524650000000001e-09, "jotun"=> 3.9527e-08, "joual"=> 6.0704920000000004e-09, "jougs"=> 3.1631140000000007e-09, "jouks"=> 1.6454825999999997e-10, "joule"=> 5.11135e-07, "jours"=> 5.456156e-07, "joust"=> 2.180454e-07, "jowar"=> 4.2515419999999995e-08, "jowed"=> 6.977102e-11, "jowls"=> 2.719132e-07, "jowly"=> 6.145132000000001e-08, "joyed"=> 5.97112e-08, "jubas"=> 4.655726e-09, "jubes"=> 4.9290819999999995e-09, "jucos"=> 2.504566e-10, "judas"=> 3.6665299999999995e-06, "judge"=> 5.1865560000000004e-05, "judgy"=> 3.094036e-08, "judos"=> 3.3672396e-10, "jugal"=> 3.1147280000000004e-08, "jugum"=> 1.0454088e-08, "juice"=> 1.680802e-05, "juicy"=> 1.81844e-06, "jujus"=> 5.151506e-09, "juked"=> 1.306082e-08, "jukes"=> 2.1757960000000002e-07, "jukus"=> 8.613192e-10, "julep"=> 1.1162748e-07, "jumar"=> 8.959124e-09, "jumbo"=> 8.818288e-07, "jumby"=> 3.2209239999999997e-09, "jumps"=> 3.846348e-06, "jumpy"=> 5.884974000000001e-07, "junco"=> 1.1420156000000001e-07, "junks"=> 1.972834e-07, "junky"=> 9.067961999999999e-08, "junta"=> 1.0188825999999999e-06, "junto"=> 2.985758e-07, "jupes"=> 7.179368000000001e-09, "jupon"=> 1.5420440000000003e-08, "jural"=> 7.549325999999999e-08, "jurat"=> 3.154756e-08, "jurel"=> 2.8584012e-09, "jures"=> 2.1849598e-09, "juror"=> 1.1541109999999998e-06, "justs"=> 9.695300000000001e-09, "jutes"=> 6.246296e-08, "jutty"=> 4.131402e-09, "juves"=> 1.7583702e-09, "juvie"=> 9.787241999999999e-08, "kaama"=> 5.106246e-09, "kabab"=> 1.3794219999999998e-08, "kabar"=> 2.085132e-08, "kabob"=> 2.793014e-08, "kacha"=> 8.450078e-08, "kacks"=> 3.5789159999999993e-10, "kadai"=> 4.801869999999999e-08, "kades"=> 3.6838340000000005e-08, "kadis"=> 1.8963119999999998e-08, "kafir"=> 1.344536e-07, "kagos"=> 7.664442e-10, "kagus"=> 5.563816400000001e-10, "kahal"=> 4.027301999999999e-08, "kaiak"=> 7.7862836e-09, "kaids"=> 2.00463e-09, "kaies"=> 3.0119438e-10, "kaifs"=> 1.1027560000000002e-09, "kaika"=> 5.0607199999999995e-08, "kaiks"=> 3.1646027999999997e-10, "kails"=> 5.493718000000001e-10, "kaims"=> 5.439692e-09, "kaing"=> 1.735336e-08, "kains"=> 1.2577855999999999e-08, "kakas"=> 1.9255368e-08, "kakis"=> 3.956364e-09, "kalam"=> 3.023856e-07, "kales"=> 5.0782440000000006e-08, "kalif"=> 5.3729546e-08, "kalis"=> 2.4654279999999997e-08, "kalpa"=> 6.734128e-08, "kamas"=> 1.1455468000000001e-08, "kames"=> 1.78835e-07, "kamik"=> 3.3960799999999997e-09, "kamis"=> 1.5724134e-08, "kamme"=> 2.1697639999999998e-09, "kanae"=> 2.64035e-08, "kanas"=> 2.217856e-08, "kandy"=> 2.7412600000000003e-07, "kaneh"=> 3.90977e-09, "kanes"=> 2.4317540000000002e-08, "kanga"=> 9.929908e-08, "kangs"=> 3.837828e-09, "kanji"=> 3.3189340000000004e-07, "kants"=> 7.657243999999999e-08, "kanzu"=> 6.444148e-09, "kaons"=> 2.40137e-08, "kapas"=> 8.483881999999999e-09, "kaphs"=> 1.9426888000000002e-10, "kapok"=> 7.520214e-08, "kapow"=> 1.7003040000000003e-08, "kappa"=> 1.10293e-06, "kapus"=> 8.581044e-09, "kaput"=> 8.968792e-08, "karas"=> 8.445882e-08, "karat"=> 1.0475982e-07, "karks"=> 1.328803e-09, "karma"=> 3.1705020000000004e-06, "karns"=> 4.849100000000001e-08, "karoo"=> 1.71706e-07, "karos"=> 1.1538244e-08, "karri"=> 5.3595020000000004e-08, "karst"=> 7.976937999999999e-07, "karsy"=> 1.3493822e-09, "karts"=> 4.801934e-08, "karzy"=> 2.70909e-10, "kasha"=> 1.3462680000000002e-07, "kasme"=> 2.47226e-09, "katal"=> 1.3907958000000001e-08, "katas"=> 2.5645e-08, "katis"=> 6.142986e-09, "katti"=> 3.7173119999999997e-08, "kaugh"=> 2.1665139999999998e-10, "kauri"=> 7.970758e-08, "kauru"=> 2.23509e-09, "kaury"=> 2.5885212e-10, "kaval"=> 2.0439240000000002e-08, "kavas"=> 5.63545e-09, "kawas"=> 2.781594e-08, "kawau"=> 7.129859999999998e-09, "kawed"=> 1.0215539999999999e-10, "kayak"=> 8.466210000000001e-07, "kayle"=> 4.8666039999999993e-08, "kayos"=> 1.7212356e-09, "kazis"=> 1.4407519999999998e-08, "kazoo"=> 5.187558e-08, "kbars"=> 4.278184e-09, "kebab"=> 2.81231e-07, "kebar"=> 2.3412299999999998e-08, "kebob"=> 4.699864e-09, "kecks"=> 6.851312e-09, "kedge"=> 5.705526e-08, "kedgy"=> 2.562362e-10, "keech"=> 1.1940478e-07, "keefs"=> 1.2390486e-10, "keeks"=> 3.714604e-09, "keels"=> 1.7013080000000002e-07, "keema"=> 2.12907e-08, "keeno"=> 5.562738e-09, "keens"=> 2.990084e-08, "keeps"=> 1.906794e-05, "keets"=> 5.096472e-09, "keeve"=> 1.2658183999999999e-08, "kefir"=> 3.453114e-07, "kehua"=> 6.090314e-09, "keirs"=> 6.147124e-10, "kelep"=> 4.0303936000000003e-10, "kelim"=> 3.064866e-08, "kells"=> 3.187172e-07, "kelly"=> 1.4396660000000003e-05, "kelps"=> 2.9555020000000002e-08, "kelpy"=> 6.54112e-09, "kelts"=> 2.541194e-08, "kelty"=> 6.276917999999999e-08, "kembo"=> 6.066822e-09, "kembs"=> 1.3588452000000002e-09, "kemps"=> 2.5143820000000002e-08, "kempt"=> 6.439426e-08, "kempy"=> 1.6943934000000002e-09, "kenaf"=> 1.2715615999999998e-07, "kench"=> 1.390042e-08, "kendo"=> 8.920346000000001e-08, "kenos"=> 3.711706e-09, "kente"=> 7.284696e-08, "kents"=> 4.7188040000000005e-08, "kepis"=> 9.939638e-09, "kerbs"=> 5.1871820000000005e-08, "kerel"=> 5.07417e-09, "kerfs"=> 6.3443040000000006e-09, "kerky"=> 1.0412037999999998e-10, "kerma"=> 1.1270982e-07, "kerne"=> 1.636012e-08, "kerns"=> 1.6360840000000001e-07, "keros"=> 4.193114e-08, "kerry"=> 2.765896e-06, "kerve"=> 8.123533999999999e-10, "kesar"=> 2.5930640000000003e-08, "kests"=> 9.688366000000001e-11, "ketas"=> 2.9149764000000003e-09, "ketch"=> 3.6748719999999997e-07, "ketes"=> 3.430932e-10, "ketol"=> 6.128211999999999e-09, "kevel"=> 8.130756e-09, "kevil"=> 5.38079e-09, "kexes"=> 3.6064860000000006e-10, "keyed"=> 1.0623931999999999e-06, "keyer"=> 8.569526000000001e-09, "khadi"=> 1.0500186e-07, "khafs"=> 1.0680644e-10, "khaki"=> 1.3224e-06, "khans"=> 1.71926e-07, "khaph"=> 4.1756517999999997e-10, "khats"=> 1.098385e-09, "khaya"=> 2.012262e-08, "khazi"=> 2.1901726e-08, "kheda"=> 8.533725999999998e-08, "kheth"=> 4.849980600000001e-10, "khets"=> 4.4388380000000004e-10, "khoja"=> 6.516086e-08, "khors"=> 5.508450000000001e-09, "khoum"=> 9.496196000000002e-10, "khuds"=> 9.398634e-10, "kiaat"=> 1.2073268e-09, "kiack"=> 1.1646692e-10, "kiang"=> 1.2156796e-07, "kibbe"=> 3.948666e-08, "kibbi"=> 3.00905e-09, "kibei"=> 2.0909706e-08, "kibes"=> 2.909344e-09, "kibla"=> 2.84828e-09, "kicks"=> 2.653872e-06, "kicky"=> 1.2661357999999998e-08, "kiddo"=> 5.141838e-07, "kiddy"=> 7.366522e-08, "kidel"=> 1.971004e-09, "kidge"=> 7.360140000000001e-10, "kiefs"=> 1.0398950000000002e-10, "kiers"=> 1.8767780000000002e-08, "kieve"=> 8.763742e-09, "kievs"=> 2.8716e-09, "kight"=> 1.6783619999999998e-08, "kikes"=> 1.947594e-08, "kikoi"=> 2.6211019999999998e-09, "kiley"=> 2.3723659999999997e-07, "kilim"=> 2.3056359999999997e-08, "kills"=> 4.6848599999999996e-06, "kilns"=> 4.4612059999999993e-07, "kilos"=> 6.09145e-07, "kilps"=> 1.3594538e-10, "kilts"=> 1.6665620000000002e-07, "kilty"=> 2.8321399999999994e-08, "kimbo"=> 2.133252e-08, "kinas"=> 3.095072e-09, "kinda"=> 2.7187699999999997e-06, "kinds"=> 4.085447999999999e-05, "kindy"=> 3.075812e-08, "kines"=> 1.8500759999999998e-08, "kings"=> 2.5802319999999997e-05, "kinin"=> 4.62228e-08, "kinks"=> 5.428832000000001e-07, "kinky"=> 6.912652e-07, "kinos"=> 1.0136844e-08, "kiore"=> 5.11171e-09, "kiosk"=> 7.710664e-07, "kipes"=> 1.3754380000000001e-09, "kippa"=> 1.0131916e-08, "kipps"=> 2.439104e-07, "kirby"=> 2.285126e-06, "kirks"=> 3.270056e-08, "kirns"=> 6.433986e-10, "kirri"=> 4.487836e-09, "kisan"=> 9.800284000000001e-08, "kissy"=> 1.1122928e-07, "kists"=> 5.20532e-09, "kited"=> 4.66726e-09, "kiter"=> 5.265212e-09, "kites"=> 5.995598e-07, "kithe"=> 3.6251719999999996e-09, "kiths"=> 1.42773e-09, "kitty"=> 6.174878000000001e-06, "kitul"=> 1.19604e-09, "kivas"=> 8.74301e-08, "kiwis"=> 1.466626e-07, "klang"=> 1.319574e-07, "klaps"=> 2.332962e-09, "klett"=> 1.279058e-07, "klick"=> 4.643468e-08, "klieg"=> 3.006046e-08, "kliks"=> 5.306641999999999e-09, "klong"=> 3.084858e-08, "kloof"=> 6.828327999999999e-08, "kluge"=> 3.087604e-07, "klutz"=> 1.1810447999999999e-07, "knack"=> 1.3468700000000002e-06, "knags"=> 3.0111740000000004e-10, "knaps"=> 1.9666722e-09, "knarl"=> 1.0285904e-09, "knars"=> 1.79453e-09, "knaur"=> 7.3818320000000006e-09, "knave"=> 8.084594000000001e-07, "knawe"=> 1.4574144000000002e-09, "knead"=> 6.970762e-07, "kneed"=> 3.29297e-07, "kneel"=> 2.342318e-06, "knees"=> 3.167436e-05, "knell"=> 4.929232000000001e-07, "knelt"=> 6.966632e-06, "knife"=> 2.597168e-05, "knish"=> 1.6241360000000002e-08, "knits"=> 2.157156e-07, "knive"=> 3.2273420000000004e-09, "knobs"=> 7.340098000000001e-07, "knock"=> 1.489196e-05, "knoll"=> 1.0855366e-06, "knops"=> 4.8376039999999995e-08, "knosp"=> 5.264722e-09, "knots"=> 4.580702e-06, "knout"=> 4.9580400000000006e-08, "knowe"=> 4.45765e-07, "known"=> 0.0002153964, "knows"=> 6.588499999999998e-05, "knubs"=> 2.4407558e-10, "knurl"=> 7.008213999999999e-09, "knurr"=> 2.4613283999999997e-09, "knurs"=> 1.6125406e-10, "knuts"=> 6.624464e-09, "koala"=> 2.2841279999999998e-07, "koans"=> 5.0795219999999997e-08, "koaps"=> 0.0, "koban"=> 1.899112e-08, "kobos"=> 5.762828e-09, "koels"=> 3.223064e-09, "koffs"=> 2.4048682e-10, "kofta"=> 4.319548e-08, "kogal"=> 9.132238e-10, "kohas"=> 1.3362045999999998e-09, "kohen"=> 1.4857579999999998e-07, "kohls"=> 2.7991020000000002e-08, "koine"=> 1.262632e-07, "kojis"=> 2.041756e-09, "kokam"=> 4.402274e-09, "kokas"=> 5.512799999999999e-09, "koker"=> 2.3240819999999998e-08, "kokra"=> 4.3104050000000005e-10, "kokum"=> 2.402928e-08, "kolas"=> 1.0349024e-08, "kolos"=> 1.512904e-08, "kombu"=> 9.782884000000001e-08, "konbu"=> 1.6702739999999997e-08, "kondo"=> 4.750554e-07, "konks"=> 5.721328e-10, "kooks"=> 3.77404e-08, "kooky"=> 9.910226e-08, "koori"=> 3.391886000000001e-08, "kopek"=> 2.78605e-08, "kophs"=> 5.67681e-11, "kopje"=> 7.07464e-08, "koppa"=> 5.6036720000000006e-09, "korai"=> 3.060796e-08, "koras"=> 6.634055999999999e-09, "korat"=> 3.616114e-08, "kores"=> 1.3066474e-08, "korma"=> 5.220892e-08, "koros"=> 4.135354e-08, "korun"=> 1.22246e-08, "korus"=> 4.0709699999999996e-08, "koses"=> 4.38636e-10, "kotch"=> 1.910734e-08, "kotos"=> 2.6536679999999998e-09, "kotow"=> 2.880984e-09, "koura"=> 2.437646e-08, "kraal"=> 2.8706159999999996e-07, "krabs"=> 8.077132e-09, "kraft"=> 1.493724e-06, "krais"=> 1.9711814000000002e-08, "krait"=> 3.687706e-08, "krang"=> 7.243153999999999e-09, "krans"=> 2.4188780000000002e-08, "kranz"=> 1.98123e-07, "kraut"=> 3.458056e-07, "krays"=> 4.500543999999999e-08, "kreep"=> 1.4414978e-08, "kreng"=> 3.839514e-09, "krewe"=> 1.0144762000000001e-07, "krill"=> 3.5670900000000005e-07, "krona"=> 6.870557999999999e-08, "krone"=> 1.899052e-07, "kroon"=> 1.285384e-07, "krubi"=> 1.6589966e-10, "krunk"=> 3.3462120000000004e-09, "ksars"=> 7.351273999999999e-10, "kubie"=> 2.774796e-08, "kudos"=> 3.082484e-07, "kudus"=> 1.8169840000000002e-08, "kudzu"=> 1.351866e-07, "kufis"=> 1.5117759999999999e-09, "kugel"=> 1.450618e-07, "kuias"=> 1.661979e-10, "kukri"=> 4.016504e-08, "kukus"=> 2.097432e-09, "kulak"=> 1.1197388000000002e-07, "kulan"=> 1.4278218e-08, "kulas"=> 1.278216e-08, "kulfi"=> 2.4943879999999996e-08, "kumis"=> 8.053658e-09, "kumys"=> 4.451888e-09, "kuris"=> 5.57758e-08, "kurre"=> 2.784268e-09, "kurta"=> 1.071388e-07, "kurus"=> 6.296226e-08, "kusso"=> 3.8657704e-10, "kutas"=> 4.406098000000001e-08, "kutch"=> 1.0392285999999999e-07, "kutis"=> 1.919154e-09, "kutus"=> 5.360714e-10, "kuzus"=> 0.0, "kvass"=> 5.3289499999999996e-08, "kvell"=> 3.837345999999999e-09, "kwela"=> 1.9234780000000003e-08, "kyack"=> 2.86699e-09, "kyaks"=> 7.983509999999999e-10, "kyang"=> 1.6447692e-08, "kyars"=> 1.906264e-09, "kyats"=> 1.4973420000000003e-08, "kybos"=> 6.335028e-10, "kydst"=> 1.2563032000000001e-10, "kyles"=> 4.6664519999999994e-08, "kylie"=> 1.2924780000000001e-06, "kylin"=> 3.7758212e-08, "kylix"=> 3.887738e-08, "kyloe"=> 3.0265799999999997e-09, "kynde"=> 7.257513999999999e-08, "kynds"=> 5.547436e-10, "kypes"=> 3.4261484e-10, "kyrie"=> 3.1804840000000004e-07, "kytes"=> 3.34509e-09, "kythe"=> 6.5296959999999995e-09, "laari"=> 1.8872920000000002e-09, "labda"=> 4.937676e-09, "label"=> 1.890204e-05, "labia"=> 7.250421999999999e-07, "labis"=> 1.5614354e-08, "labor"=> 6.769182e-05, "labra"=> 3.3082839999999995e-08, "laced"=> 3.014628e-06, "lacer"=> 9.91508e-09, "laces"=> 1.0111748000000002e-06, "lacet"=> 3.756158e-09, "lacey"=> 2.609754e-06, "lacks"=> 6.965458e-06, "laddy"=> 5.002318e-08, "laded"=> 2.5135279999999996e-08, "laden"=> 6.646250000000001e-06, "lader"=> 1.19325e-07, "lades"=> 2.1834806e-08, "ladle"=> 8.182833999999999e-07, "laers"=> 7.206955999999999e-10, "laevo"=> 7.697952e-09, "lagan"=> 7.498146e-08, "lager"=> 8.165786e-07, "lahal"=> 1.4319628000000002e-09, "lahar"=> 4.957808e-08, "laich"=> 6.138723999999999e-09, "laics"=> 8.510702e-09, "laids"=> 2.664034e-09, "laigh"=> 9.346208000000002e-09, "laika"=> 9.138801999999999e-08, "laiks"=> 3.859264e-09, "laird"=> 1.944042e-06, "lairs"=> 1.738732e-07, "lairy"=> 1.0223445999999998e-08, "laith"=> 8.3029e-08, "laity"=> 1.3455520000000002e-06, "laked"=> 4.2135820000000005e-09, "laker"=> 1.409498e-07, "lakes"=> 8.811974e-06, "lakhs"=> 4.249912e-07, "lakin"=> 8.062484e-08, "laksa"=> 4.7649580000000005e-08, "laldy"=> 1.0383e-09, "lalls"=> 2.936452e-09, "lamas"=> 3.571704e-07, "lambs"=> 2.319772e-06, "lamby"=> 1.5690060000000003e-08, "lamed"=> 1.6060759999999998e-07, "lamer"=> 6.791746e-08, "lames"=> 4.3915180000000004e-08, "lamia"=> 3.0049320000000003e-07, "lammy"=> 2.7540717999999998e-08, "lamps"=> 6.222458e-06, "lanai"=> 1.350242e-07, "lanas"=> 1.309536e-08, "lance"=> 5.786344e-06, "lanch"=> 5.584296e-09, "lande"=> 7.445809999999999e-07, "lands"=> 2.3405059999999998e-05, "lanes"=> 3.536738e-06, "lanks"=> 3.635538e-09, "lanky"=> 9.421441999999999e-07, "lants"=> 1.8760060000000003e-08, "lapel"=> 5.881994e-07, "lapin"=> 1.2307676000000001e-07, "lapis"=> 4.7032719999999996e-07, "lapje"=> 2.3039583999999998e-10, "lapse"=> 2.905818e-06, "larch"=> 3.466828e-07, "lards"=> 7.59737e-09, "lardy"=> 8.42271e-08, "laree"=> 1.197631e-08, "lares"=> 1.4481212e-07, "large"=> 0.0002670708, "largo"=> 6.423178e-07, "laris"=> 2.481928e-08, "larks"=> 3.499806e-07, "larky"=> 1.392486e-08, "larns"=> 2.8326499999999997e-09, "larnt"=> 1.1967512e-08, "larum"=> 1.650254e-08, "larva"=> 9.263142000000001e-07, "lased"=> 8.506587999999999e-09, "laser"=> 1.5919980000000002e-05, "lases"=> 4.79177e-09, "lassi"=> 9.395581999999999e-08, "lasso"=> 7.817328e-07, "lassu"=> 7.860513999999999e-10, "lassy"=> 9.442422e-09, "lasts"=> 3.673106e-06, "latah"=> 2.27423e-08, "latch"=> 2.662888e-06, "lated"=> 1.878574e-07, "laten"=> 9.453832e-08, "later"=> 0.0002644112, "latex"=> 1.803604e-06, "lathe"=> 5.315358e-07, "lathi"=> 4.350858e-08, "laths"=> 1.259266e-07, "lathy"=> 9.40058e-09, "latke"=> 1.2107068000000002e-08, "latte"=> 7.806619999999999e-07, "latus"=> 7.658326e-08, "lauan"=> 6.222226e-09, "lauch"=> 1.2506672e-08, "lauds"=> 1.8869960000000002e-07, "laufs"=> 1.1548486e-08, "laugh"=> 4.033438e-05, "laund"=> 5.916594000000001e-09, "laura"=> 1.603346e-05, "laval"=> 7.138352e-07, "lavas"=> 2.296134e-07, "laved"=> 1.1538492e-07, "laver"=> 3.192008e-07, "laves"=> 9.449018e-08, "lavra"=> 6.089012e-08, "lavvy"=> 2.1178320000000002e-08, "lawed"=> 7.003703999999999e-09, "lawer"=> 1.3660924e-08, "lawin"=> 8.681856e-09, "lawks"=> 1.0488787999999999e-08, "lawns"=> 1.6237799999999999e-06, "lawny"=> 8.013146e-09, "laxed"=> 5.190047999999999e-09, "laxer"=> 5.244589999999999e-08, "laxes"=> 9.268514000000001e-10, "laxly"=> 1.695958e-08, "layed"=> 8.985722000000001e-08, "layer"=> 4.4861e-05, "layin"=> 1.3828881999999998e-07, "layup"=> 1.3841420000000002e-07, "lazar"=> 5.632373999999999e-07, "lazed"=> 5.713768000000001e-08, "lazes"=> 9.098712e-09, "lazos"=> 2.3911599999999998e-08, "lazzi"=> 2.3382239999999998e-08, "lazzo"=> 5.2592859999999995e-09, "leach"=> 1.7500440000000001e-06, "leads"=> 4.654826e-05, "leady"=> 1.4265542e-08, "leafs"=> 3.5200479999999997e-07, "leafy"=> 2.0455459999999997e-06, "leaks"=> 2.3332819999999997e-06, "leaky"=> 9.803794e-07, "leams"=> 1.7474306e-09, "leans"=> 4.2040380000000005e-06, "leant"=> 2.009284e-06, "leany"=> 3.3203959999999998e-09, "leaps"=> 2.33972e-06, "leapt"=> 4.883072e-06, "leare"=> 3.3713816e-09, "learn"=> 9.158542e-05, "lears"=> 9.156916e-08, "leary"=> 8.861519999999999e-07, "lease"=> 1.0117356e-05, "leash"=> 2.36936e-06, "least"=> 0.0002270318, "leats"=> 4.745164000000001e-09, "leave"=> 0.00015926379999999998, "leavy"=> 1.1589700000000001e-07, "leaze"=> 5.5094640000000006e-09, "leben"=> 1.2629008e-06, "leccy"=> 2.5322980000000002e-09, "ledes"=> 8.08018e-09, "ledge"=> 3.62803e-06, "ledgy"=> 3.1526400000000004e-09, "ledum"=> 1.666514e-08, "leear"=> 1.969974e-09, "leech"=> 1.075124e-06, "leeks"=> 6.57032e-07, "leeps"=> 1.754736e-09, "leers"=> 1.0063572000000002e-07, "leery"=> 3.2902799999999997e-07, "leese"=> 1.9370760000000003e-07, "leets"=> 1.2142682e-08, "leeze"=> 1.168792e-08, "lefte"=> 2.2382260000000002e-08, "lefts"=> 7.683540000000001e-08, "lefty"=> 5.867098e-07, "legal"=> 0.00011498920000000001, "leger"=> 3.496348e-07, "leges"=> 1.9654059999999996e-07, "legge"=> 6.691614e-07, "leggo"=> 5.587792e-08, "leggy"=> 1.8012779999999997e-07, "legit"=> 5.036492e-07, "lehrs"=> 1.0275704e-08, "lehua"=> 2.075918e-08, "leirs"=> 4.566634e-09, "leish"=> 1.7137396e-08, "leman"=> 1.8665299999999997e-07, "lemed"=> 9.715550000000002e-10, "lemel"=> 6.405488e-09, "lemes"=> 9.696584e-09, "lemma"=> 5.081062e-06, "lemme"=> 3.7129e-07, "lemon"=> 1.14697e-05, "lemur"=> 1.9627620000000004e-07, "lends"=> 2.3116760000000004e-06, "lenes"=> 5.872019999999999e-09, "lengs"=> 8.28605e-10, "lenis"=> 4.1341459999999994e-08, "lenos"=> 1.532787e-08, "lense"=> 3.3588480000000004e-08, "lenti"=> 1.8394379999999998e-08, "lento"=> 7.815012e-08, "leone"=> 3.1165379999999995e-06, "leper"=> 8.992856e-07, "lepid"=> 4.79617e-09, "lepra"=> 6.419596e-08, "lepta"=> 2.2307403999999996e-08, "lered"=> 2.527894e-09, "leres"=> 2.9269412e-09, "lerps"=> 1.7420980000000002e-09, "lesbo"=> 2.2008239999999998e-08, "leses"=> 7.272984000000001e-10, "lests"=> 6.861358e-10, "letch"=> 3.400132e-08, "lethe"=> 2.2822860000000003e-07, "letup"=> 3.474592e-08, "leuch"=> 3.265108e-09, "leuco"=> 2.916634e-08, "leuds"=> 3.2339646e-10, "leugh"=> 1.8198893999999998e-09, "levas"=> 1.1473771999999999e-08, "levee"=> 6.508812000000001e-07, "level"=> 0.0002404882, "lever"=> 4.009838e-06, "leves"=> 2.36001e-08, "levin"=> 3.6981800000000007e-06, "levis"=> 2.59101e-07, "lewis"=> 2.235344e-05, "lexes"=> 2.04723e-09, "lexis"=> 3.25726e-06, "lezes"=> 1.4721564e-10, "lezza"=> 2.4760560000000006e-09, "lezzy"=> 2.9625280000000007e-09, "liana"=> 4.063782e-07, "liane"=> 1.697354e-07, "liang"=> 4.029902000000001e-06, "liard"=> 3.9280080000000004e-08, "liars"=> 1.125882e-06, "liart"=> 1.1413394e-09, "libel"=> 1.56705e-06, "liber"=> 1.3578240000000002e-06, "libra"=> 6.198998000000001e-07, "libri"=> 6.88367e-07, "lichi"=> 6.704132000000001e-09, "licht"=> 5.334888e-07, "licit"=> 2.7588300000000004e-07, "licks"=> 1.0355633999999999e-06, "lidar"=> 7.477904000000001e-07, "lidos"=> 8.33779e-09, "liefs"=> 9.144881999999999e-09, "liege"=> 6.101958e-07, "liens"=> 7.401454e-07, "liers"=> 3.883190000000001e-08, "lieus"=> 1.9325592e-09, "lieve"=> 9.559756e-08, "lifer"=> 1.1839319999999998e-07, "lifes"=> 1.666628e-07, "lifts"=> 4.554566000000001e-06, "ligan"=> 1.0627100000000002e-08, "liger"=> 2.994398e-08, "ligge"=> 3.3318996e-08, "light"=> 0.00023950259999999999, "ligne"=> 1.8236679999999997e-07, "liked"=> 4.3221259999999996e-05, "liken"=> 4.205257999999999e-07, "liker"=> 9.068149999999999e-08, "likes"=> 1.536212e-05, "likin"=> 4.807556e-08, "lilac"=> 1.204662e-06, "lills"=> 1.494818e-09, "lilos"=> 6.387894000000001e-09, "lilts"=> 1.280866e-08, "liman"=> 1.1185484e-07, "limas"=> 2.6779560000000004e-08, "limax"=> 1.0746356e-08, "limba"=> 5.2194279999999994e-08, "limbi"=> 7.0682640000000014e-09, "limbo"=> 1.1427340000000002e-06, "limbs"=> 1.107668e-05, "limby"=> 1.4611972e-09, "limed"=> 5.165956e-08, "limen"=> 5.7482339999999994e-08, "limes"=> 6.550368e-07, "limey"=> 6.369864e-08, "limit"=> 4.151956e-05, "limma"=> 1.1189566e-08, "limns"=> 1.9691499999999998e-08, "limos"=> 1.228028e-07, "limpa"=> 1.1230908e-08, "limps"=> 1.550908e-07, "linac"=> 1.1318182000000001e-07, "linch"=> 2.430172e-08, "linds"=> 3.01216e-08, "lindy"=> 7.540286000000001e-07, "lined"=> 1.3834860000000001e-05, "linen"=> 7.215186e-06, "liner"=> 2.2117920000000004e-06, "lines"=> 8.585224000000002e-05, "liney"=> 8.317188e-09, "linga"=> 1.0712969999999999e-07, "lingo"=> 5.366899999999999e-07, "lings"=> 5.61169e-08, "lingy"=> 2.105842e-09, "linin"=> 1.3067665999999998e-08, "links"=> 2.7837900000000003e-05, "linky"=> 7.647969999999999e-09, "linns"=> 2.5013806e-08, "linny"=> 8.55568e-08, "linos"=> 2.5012580000000005e-08, "lints"=> 1.594136e-08, "linty"=> 9.527738e-09, "linum"=> 6.519040000000001e-08, "linux"=> 3.137254e-06, "lions"=> 4.99848e-06, "lipas"=> 2.8253744e-09, "lipes"=> 3.1424040000000003e-09, "lipid"=> 7.358412e-06, "lipin"=> 1.454112e-08, "lipos"=> 6.207902e-09, "lippy"=> 7.365352000000001e-08, "liras"=> 6.884864e-08, "lirks"=> 1.0249806000000001e-10, "lirot"=> 1.6949802e-09, "lisks"=> 2.83909e-10, "lisle"=> 4.6550640000000006e-07, "lisps"=> 2.4279559999999995e-08, "lists"=> 1.732126e-05, "litai"=> 1.5655499999999997e-08, "litas"=> 1.5113768e-08, "lited"=> 1.5994040000000003e-09, "liter"=> 1.316998e-06, "lites"=> 8.893974e-08, "lithe"=> 1.0323751999999999e-06, "litho"=> 6.864529999999999e-08, "liths"=> 3.149208e-09, "litre"=> 1.0256778e-06, "lived"=> 7.943068000000001e-05, "liven"=> 1.912242e-07, "liver"=> 2.454614e-05, "lives"=> 0.0001078872, "livid"=> 1.353606e-06, "livor"=> 2.7038420000000002e-08, "livre"=> 8.735228000000001e-07, "llama"=> 4.641506e-07, "llano"=> 1.7326960000000002e-07, "loach"=> 1.5400580000000003e-07, "loads"=> 9.099896e-06, "loafs"=> 2.0863300000000003e-08, "loams"=> 6.609012000000002e-08, "loamy"=> 2.1447540000000001e-07, "loans"=> 1.527082e-05, "loast"=> 7.879046000000001e-10, "loath"=> 8.574782000000001e-07, "loave"=> 6.37734e-10, "lobar"=> 3.501446e-07, "lobby"=> 8.569918e-06, "lobed"=> 3.79281e-07, "lobes"=> 2.379522e-06, "lobos"=> 1.6753619999999998e-07, "lobus"=> 1.6542605999999998e-08, "local"=> 0.0001914902, "loche"=> 1.525084e-08, "lochs"=> 1.2767059999999998e-07, "locie"=> 4.837048e-10, "locis"=> 8.300323999999999e-08, "locks"=> 5.904186e-06, "locos"=> 1.2457134e-07, "locum"=> 2.4829999999999997e-07, "locus"=> 5.141299999999999e-06, "loden"=> 4.7621660000000005e-08, "lodes"=> 4.6761560000000004e-08, "lodge"=> 9.598506e-06, "loess"=> 4.4427060000000005e-07, "lofts"=> 2.2126639999999999e-07, "lofty"=> 4.62448e-06, "logan"=> 9.202582e-06, "loges"=> 3.4882320000000004e-08, "loggy"=> 4.2976979999999995e-09, "logia"=> 5.7053519999999995e-08, "logic"=> 3.610534e-05, "logie"=> 9.825898e-08, "login"=> 1.387732e-06, "logoi"=> 1.6362900000000002e-07, "logon"=> 2.610756e-07, "logos"=> 3.30757e-06, "lohan"=> 8.201458e-08, "loids"=> 3.819012e-09, "loins"=> 1.2422464e-06, "loipe"=> 2.5244244e-10, "loirs"=> 4.288942e-10, "lokes"=> 6.42946e-09, "lolls"=> 4.9713840000000006e-08, "lolly"=> 3.511142e-07, "lolog"=> 6.498704e-10, "lomas"=> 2.6061500000000003e-07, "lomed"=> 2.724133e-10, "lomes"=> 8.378426000000001e-10, "loner"=> 6.107608e-07, "longa"=> 2.401622e-07, "longe"=> 2.9934260000000004e-07, "longs"=> 1.0735522e-06, "looby"=> 3.599534e-08, "looed"=> 1.5974942e-08, "looey"=> 1.4768142000000003e-08, "loofa"=> 1.1405384e-08, "loofs"=> 1.0391706e-08, "looie"=> 7.811896e-09, "looks"=> 7.624389999999999e-05, "looky"=> 1.1127917999999999e-07, "looms"=> 1.392602e-06, "loons"=> 1.745478e-07, "loony"=> 2.6928299999999996e-07, "loops"=> 5.407894e-06, "loopy"=> 2.297622e-07, "loord"=> 1.165596e-09, "loose"=> 2.539488e-05, "loots"=> 3.809414e-08, "loped"=> 3.06129e-07, "loper"=> 5.9092119999999996e-08, "lopes"=> 7.366642e-07, "loppy"=> 5.784802e-09, "loral"=> 3.965062e-08, "loran"=> 1.0872658e-07, "lords"=> 9.834170000000001e-06, "lordy"=> 1.762994e-07, "lorel"=> 2.7634595999999996e-08, "lores"=> 9.943034e-08, "loric"=> 4.82982e-08, "loris"=> 1.8726880000000002e-07, "lorry"=> 1.46778e-06, "losed"=> 1.0931304e-08, "losel"=> 1.647064e-08, "losen"=> 4.143658e-08, "loser"=> 2.284684e-06, "loses"=> 6.46192e-06, "lossy"=> 3.325242e-07, "lotah"=> 1.7866556000000002e-09, "lotas"=> 2.887686e-09, "lotes"=> 7.326572000000001e-09, "lotic"=> 3.7247840000000006e-08, "lotos"=> 7.929336e-08, "lotsa"=> 4.674218e-08, "lotta"=> 4.46228e-07, "lotte"=> 4.292386e-07, "lotto"=> 2.38124e-07, "lotus"=> 2.98476e-06, "loued"=> 8.566984e-08, "lough"=> 4.108834e-07, "louie"=> 1.299108e-06, "louis"=> 3.020646e-05, "louma"=> 2.2034339999999997e-09, "lound"=> 5.195638000000001e-09, "louns"=> 1.4657322e-09, "loupe"=> 9.704320000000001e-08, "loups"=> 5.321504e-08, "loure"=> 5.616496e-09, "lours"=> 7.207898e-09, "loury"=> 4.080366e-08, "louse"=> 4.854248e-07, "lousy"=> 1.4343859999999998e-06, "louts"=> 1.1227361999999999e-07, "lovat"=> 1.607716e-07, "loved"=> 7.140768000000001e-05, "lover"=> 1.74864e-05, "loves"=> 1.7915299999999998e-05, "lovey"=> 3.059298e-07, "lovie"=> 8.883558e-08, "lowan"=> 1.0478681999999999e-08, "lowed"=> 2.20841e-07, "lower"=> 0.00012339759999999999, "lowes"=> 1.297656e-07, "lowly"=> 2.2410780000000003e-06, "lownd"=> 1.1321555e-08, "lowne"=> 2.3830140000000005e-09, "lowns"=> 2.782722e-09, "lowps"=> 6.168312e-11, "lowry"=> 9.212169999999999e-07, "lowse"=> 6.727115999999999e-09, "lowts"=> 3.2353859999999997e-10, "loxed"=> 1.3669109999999998e-10, "loxes"=> 1.33507054e-09, "loyal"=> 1.0205768e-05, "lozen"=> 2.1094149999999998e-08, "luach"=> 5.439252e-09, "luaus"=> 7.973225999999999e-09, "lubed"=> 8.33305e-08, "lubes"=> 2.2457759999999998e-08, "lubra"=> 1.0926548e-08, "luces"=> 5.075992e-08, "lucid"=> 1.777216e-06, "lucks"=> 2.71576e-08, "lucky"=> 2.3218099999999997e-05, "lucre"=> 1.775696e-07, "ludes"=> 2.540736e-08, "ludic"=> 3.176816e-07, "ludos"=> 7.43873e-09, "luffa"=> 3.510748e-08, "luffs"=> 7.117072e-09, "luged"=> 5.232382e-10, "luger"=> 2.467432e-07, "luges"=> 3.437122e-09, "lulls"=> 1.8278520000000002e-07, "lulus"=> 1.9298180000000003e-08, "lumas"=> 8.507418e-09, "lumbi"=> 3.452533e-09, "lumen"=> 2.504334e-06, "lumme"=> 2.2661979999999998e-08, "lummy"=> 9.919058000000001e-09, "lumps"=> 1.6449219999999999e-06, "lumpy"=> 8.521497999999999e-07, "lunar"=> 3.3237359999999998e-06, "lunas"=> 3.1748379999999996e-08, "lunch"=> 2.981172e-05, "lunes"=> 6.300002000000001e-08, "lunet"=> 4.687469999999999e-09, "lunge"=> 1.0270253999999999e-06, "lungi"=> 7.007144e-08, "lungs"=> 1.341858e-05, "lunks"=> 3.0172899999999996e-09, "lunts"=> 1.364958e-08, "lupin"=> 5.370976e-07, "lupus"=> 2.162536e-06, "lurch"=> 1.154646e-06, "lured"=> 1.701406e-06, "lurer"=> 4.60463e-09, "lures"=> 5.564476e-07, "lurex"=> 1.2052420000000001e-08, "lurgi"=> 2.7645e-08, "lurgy"=> 5.334082e-09, "lurid"=> 1.0734428e-06, "lurks"=> 6.435477999999999e-07, "lurry"=> 2.913134e-09, "lurve"=> 9.482784e-09, "luser"=> 1.123946e-09, "lushy"=> 4.182204e-09, "lusks"=> 1.1230196e-09, "lusts"=> 8.181356e-07, "lusty"=> 7.591444e-07, "lusus"=> 3.801614e-08, "lutea"=> 1.1215282e-07, "luted"=> 1.517042e-08, "luter"=> 2.1233820000000002e-08, "lutes"=> 1.4545079999999999e-07, "luvvy"=> 3.516564e-09, "luxed"=> 3.2823439999999997e-10, "luxer"=> 6.507412e-10, "luxes"=> 5.398582e-09, "lweis"=> 2.8985659999999997e-10, "lyams"=> 9.388438e-11, "lyard"=> 2.254026e-09, "lyart"=> 3.4890293999999997e-09, "lyase"=> 2.253526e-07, "lycea"=> 3.9224298e-09, "lycee"=> 1.679056e-08, "lycra"=> 1.6989820000000001e-07, "lying"=> 4.138268e-05, "lymes"=> 4.873238e-09, "lymph"=> 5.657232000000001e-06, "lynch"=> 4.53655e-06, "lynes"=> 6.271784000000001e-08, "lyres"=> 1.1091008e-07, "lyric"=> 3.000406e-06, "lysed"=> 1.187306e-07, "lyses"=> 2.91409e-08, "lysin"=> 1.9476059999999998e-08, "lysis"=> 9.59234e-07, "lysol"=> 6.163874000000001e-08, "lyssa"=> 2.39128e-07, "lyted"=> 6.615058e-11, "lytes"=> 1.2062432000000001e-08, "lythe"=> 1.88369e-08, "lytic"=> 4.4629899999999994e-07, "lytta"=> 6.969324e-09, "maaed"=> 8.511626e-11, "maare"=> 2.2986880000000002e-09, "maars"=> 1.4228984e-08, "mabes"=> 1.6491684e-09, "macas"=> 1.317452e-08, "macaw"=> 1.5314260000000002e-07, "maced"=> 1.5948260000000002e-08, "macer"=> 6.956312e-08, "maces"=> 1.064307e-07, "mache"=> 1.5322251999999997e-07, "machi"=> 1.6979919999999998e-07, "macho"=> 9.971148e-07, "machs"=> 1.2194512000000001e-08, "macks"=> 3.624056e-08, "macle"=> 3.8006599999999996e-09, "macon"=> 8.661174e-07, "macro"=> 7.730394e-06, "madam"=> 7.3336720000000015e-06, "madge"=> 1.0305236000000002e-06, "madid"=> 1.9148769999999998e-09, "madly"=> 1.964032e-06, "madre"=> 1.0528476e-06, "maerl"=> 9.668798e-09, "mafia"=> 2.386932e-06, "mafic"=> 2.5131579999999996e-07, "mages"=> 8.538106e-07, "maggs"=> 1.9087659999999997e-07, "magic"=> 3.466081999999999e-05, "magma"=> 1.1886387999999999e-06, "magot"=> 2.0755992e-08, "magus"=> 5.781555999999999e-07, "mahoe"=> 1.2181121999999998e-08, "mahua"=> 3.9847459999999994e-08, "mahwa"=> 1.439754e-09, "maids"=> 2.87659e-06, "maiko"=> 3.325624e-08, "maiks"=> 1.71732706e-09, "maile"=> 5.012328e-08, "maill"=> 4.0934380000000005e-09, "mails"=> 2.026958e-06, "maims"=> 2.414024e-08, "mains"=> 2.094952e-06, "maire"=> 2.945538e-07, "mairs"=> 3.8235659999999997e-08, "maise"=> 2.0451860000000002e-08, "maist"=> 7.510614e-08, "maize"=> 4.285174e-06, "major"=> 0.000145607, "makar"=> 1.9901639999999998e-07, "maker"=> 9.874465999999999e-06, "makes"=> 0.0001413634, "makis"=> 1.944806e-08, "makos"=> 1.2156698e-08, "malam"=> 9.987513999999999e-08, "malar"=> 2.491686e-07, "malas"=> 8.123586e-08, "malax"=> 9.080321999999999e-10, "males"=> 1.6366079999999998e-05, "malic"=> 2.079938e-07, "malik"=> 2.49149e-06, "malis"=> 5.9426379999999994e-08, "malls"=> 1.2722660000000002e-06, "malms"=> 3.05635e-09, "malmy"=> 2.627438e-10, "malts"=> 1.5391939999999999e-07, "malty"=> 8.026945999999999e-08, "malus"=> 1.804724e-07, "malva"=> 5.6782660000000004e-08, "malwa"=> 8.705836e-08, "mamas"=> 2.3278379999999996e-07, "mamba"=> 1.4244759999999998e-07, "mambo"=> 2.352804e-07, "mamee"=> 3.3792519999999997e-09, "mamey"=> 1.947418e-08, "mamie"=> 6.438680000000001e-07, "mamma"=> 4.131682e-06, "mammy"=> 9.143024000000001e-07, "manas"=> 2.649682e-07, "manat"=> 3.2908959999999994e-08, "mandi"=> 2.1634539999999996e-07, "maneb"=> 1.5634442e-08, "maned"=> 9.125288000000002e-08, "maneh"=> 1.2582674e-08, "manes"=> 5.07676e-07, "manet"=> 7.139548e-07, "manga"=> 9.490520000000001e-07, "mange"=> 6.729114e-07, "mango"=> 2.3096259999999997e-06, "mangs"=> 6.783188e-09, "mangy"=> 3.175304e-07, "mania"=> 2.1181799999999998e-06, "manic"=> 1.797946e-06, "manis"=> 1.1128841999999999e-07, "manky"=> 3.692344e-08, "manly"=> 3.1156760000000003e-06, "manna"=> 1.1535926000000001e-06, "manor"=> 4.895122e-06, "manos"=> 3.718578e-07, "manse"=> 4.695917999999999e-07, "manta"=> 2.4533439999999996e-07, "manto"=> 1.3227798e-07, "manty"=> 1.2379584e-08, "manul"=> 7.09149e-09, "manus"=> 5.051642e-07, "mapau"=> 2.3197859999999998e-10, "maple"=> 4.425158e-06, "maqui"=> 1.8203403999999997e-08, "marae"=> 1.707894e-07, "marah"=> 1.616964e-07, "maras"=> 9.385672000000001e-08, "march"=> 8.400524e-05, "marcs"=> 2.0152239999999998e-08, "mardy"=> 2.1801339999999997e-08, "mares"=> 9.041752000000001e-07, "marge"=> 9.520078000000001e-07, "margs"=> 3.8077160000000005e-09, "maria"=> 2.1137679999999998e-05, "marid"=> 3.93424e-08, "marka"=> 7.333396e-08, "marks"=> 2.6835960000000003e-05, "marle"=> 6.935492e-08, "marls"=> 7.725384e-08, "marly"=> 2.93807e-07, "marms"=> 3.724524e-09, "maron"=> 2.0940699999999996e-07, "maror"=> 1.059444e-08, "marra"=> 1.9356639999999997e-07, "marri"=> 5.865992e-08, "marry"=> 2.7549300000000004e-05, "marse"=> 2.3749214e-07, "marsh"=> 6.1221260000000005e-06, "marts"=> 1.6277719999999999e-07, "marvy"=> 7.722696000000001e-09, "masas"=> 2.9636800000000005e-08, "mased"=> 1.1935324e-09, "maser"=> 1.5331640000000002e-07, "mases"=> 5.7178240000000005e-09, "mashy"=> 6.685492e-09, "masks"=> 4.908582e-06, "mason"=> 1.202212e-05, "massa"=> 1.2411308e-06, "masse"=> 1.173986e-06, "massy"=> 2.52271e-07, "masts"=> 1.2370698000000001e-06, "masty"=> 6.761566e-09, "masus"=> 2.4622715999999997e-09, "matai"=> 5.665518e-08, "match"=> 3.460136e-05, "mated"=> 9.649452e-07, "mater"=> 6.24897e-06, "mates"=> 4.630295999999999e-06, "matey"=> 1.4104299999999997e-07, "maths"=> 1.5414520000000001e-06, "matin"=> 3.634578e-07, "matlo"=> 6.069844600000001e-09, "matte"=> 6.445098e-07, "matts"=> 3.3965339999999996e-08, "matza"=> 1.1794841999999999e-07, "matzo"=> 1.06259e-07, "mauby"=> 3.5230840000000002e-09, "mauds"=> 1.983732e-09, "mauls"=> 3.096024e-08, "maund"=> 4.28022e-08, "mauri"=> 1.5369340000000002e-07, "mausy"=> 1.5654464e-10, "mauts"=> 1.1831815999999998e-10, "mauve"=> 5.67205e-07, "mauzy"=> 1.8619040000000002e-08, "maven"=> 4.156518e-07, "mavie"=> 4.027328e-09, "mavin"=> 2.6277799999999998e-08, "mavis"=> 1.0413328e-06, "mawed"=> 2.310346e-09, "mawks"=> 4.973888e-10, "mawky"=> 3.12921e-10, "mawns"=> 1.313358e-10, "mawrs"=> 5.367946e-11, "maxed"=> 1.4344259999999997e-07, "maxes"=> 1.490928e-08, "maxim"=> 3.022958e-06, "maxis"=> 4.56758e-08, "mayan"=> 1.0214842e-06, "mayas"=> 1.56198e-07, "maybe"=> 0.00013772699999999998, "mayed"=> 1.6168160000000002e-09, "mayor"=> 1.365494e-05, "mayos"=> 1.34571e-08, "mayst"=> 2.9212820000000004e-07, "mazed"=> 3.989976e-08, "mazer"=> 1.1000583999999999e-07, "mazes"=> 3.963858e-07, "mazey"=> 4.148486e-08, "mazut"=> 2.7729699999999998e-09, "mbira"=> 5.4612600000000006e-08, "meads"=> 1.703214e-07, "meals"=> 1.3218639999999999e-05, "mealy"=> 2.591208e-07, "meane"=> 1.2527548e-07, "means"=> 0.0002227292, "meant"=> 9.322365999999999e-05, "meany"=> 1.4224772e-07, "meare"=> 9.880286e-09, "mease"=> 3.290028e-08, "meath"=> 2.456218e-07, "meats"=> 2.5454960000000003e-06, "meaty"=> 8.035318e-07, "mebos"=> 1.2231696000000001e-09, "mecca"=> 2.2211e-06, "mechs"=> 1.6019654e-07, "mecks"=> 1.0136026e-09, "medal"=> 4.475189999999999e-06, "media"=> 0.00011050940000000001, "medic"=> 1.0774896e-06, "medii"=> 6.945012e-08, "medle"=> 9.531228e-09, "meeds"=> 9.492172000000002e-09, "meers"=> 4.807274e-08, "meets"=> 1.204268e-05, "meffs"=> 2.1751444e-10, "meins"=> 4.38561e-08, "meint"=> 3.352322e-08, "meiny"=> 4.2835739999999995e-09, "meith"=> 1.4943488e-09, "mekka"=> 3.640346e-08, "melas"=> 1.72287e-07, "melba"=> 2.868182e-07, "melds"=> 9.686742000000002e-08, "melee"=> 6.812314e-07, "melic"=> 2.097008e-08, "melik"=> 5.2398659999999996e-08, "mells"=> 1.435786e-08, "melon"=> 1.239532e-06, "melts"=> 1.8979580000000002e-06, "melty"=> 6.217016e-08, "memes"=> 6.510183999999999e-07, "memos"=> 9.23604e-07, "menad"=> 2.126564e-09, "mends"=> 1.0717024000000001e-07, "mened"=> 1.0796702e-09, "menes"=> 6.781942e-08, "menge"=> 2.6099076999999998e-06, "mengs"=> 3.148922e-08, "mensa"=> 1.483006e-07, "mense"=> 2.6790859999999997e-07, "mensh"=> 5.145194e-09, "menta"=> 3.56863e-08, "mento"=> 4.62977e-08, "menus"=> 2.354916e-06, "meous"=> 1.6099381999999999e-10, "meows"=> 6.768626e-08, "merch"=> 7.790396000000001e-08, "mercs"=> 9.156440000000001e-08, "mercy"=> 2.133824e-05, "merde"=> 1.264276e-07, "mered"=> 3.707978e-08, "merel"=> 2.34384e-08, "merer"=> 4.068924e-09, "meres"=> 9.516758000000001e-08, "merge"=> 4.099505999999999e-06, "meril"=> 7.750292e-09, "meris"=> 4.576242e-08, "merit"=> 9.969728e-06, "merks"=> 2.6366679999999998e-08, "merle"=> 1.0900826e-06, "merls"=> 7.445626400000001e-10, "merry"=> 7.751222e-06, "merse"=> 1.532584e-08, "mesal"=> 1.4235328000000002e-08, "mesas"=> 1.763408e-07, "mesel"=> 7.182828000000001e-09, "meses"=> 1.1405354e-07, "meshy"=> 2.6323e-09, "mesic"=> 1.0585036000000001e-07, "mesne"=> 4.6331160000000004e-08, "meson"=> 1.980136e-07, "messy"=> 4.3781739999999996e-06, "mesto"=> 6.16373e-08, "metal"=> 5.202783999999999e-05, "meted"=> 6.164494e-07, "meter"=> 7.274317999999999e-06, "metes"=> 1.030607e-07, "metho"=> 1.589528e-08, "meths"=> 1.3655552e-08, "metic"=> 4.91782e-08, "metif"=> 4.525282e-10, "metis"=> 3.4720019999999997e-07, "metol"=> 9.789194000000001e-09, "metre"=> 2.4400920000000004e-06, "metro"=> 3.661368e-06, "meuse"=> 4.68643e-07, "meved"=> 8.155756e-10, "meves"=> 6.395536e-09, "mewed"=> 1.373196e-07, "mewls"=> 2.78195e-08, "meynt"=> 7.677639999999999e-10, "mezes"=> 1.836004e-08, "mezze"=> 4.0153420000000004e-08, "mezzo"=> 4.201718e-07, "mhorr"=> 9.203349999999999e-10, "miaou"=> 5.386127999999999e-09, "miaow"=> 3.8896279999999996e-08, "miasm"=> 1.1830542e-08, "miaul"=> 4.5882764e-10, "micas"=> 8.094799999999999e-08, "miche"=> 4.078776e-08, "micht"=> 4.3559800000000003e-08, "micks"=> 3.213988e-08, "micky"=> 5.310832e-07, "micos"=> 1.0568264e-08, "micra"=> 3.368066e-08, "micro"=> 1.2767060000000002e-05, "middy"=> 4.985424e-08, "midge"=> 4.2916999999999997e-07, "midgy"=> 6.232594e-10, "midis"=> 7.537345400000002e-08, "midst"=> 1.67161e-05, "miens"=> 1.326368e-08, "mieve"=> 2.6561766e-10, "miffs"=> 2.553418e-09, "miffy"=> 1.6535580000000002e-08, "mifty"=> 2.1148584000000001e-10, "miggs"=> 1.4597420000000002e-07, "might"=> 0.000456475, "mihas"=> 8.680468e-09, "mihis"=> 2.7889378000000004e-10, "miked"=> 2.518982e-08, "mikes"=> 1.12466e-07, "mikra"=> 2.295478e-08, "mikva"=> 1.9834320000000004e-08, "milch"=> 2.613714e-07, "milds"=> 1.0274088000000001e-08, "miler"=> 7.275598000000001e-08, "miles"=> 6.508219999999998e-05, "milfs"=> 9.136030000000001e-09, "milia"=> 1.315369e-07, "milko"=> 1.0630272e-08, "milks"=> 2.8997000000000005e-07, "milky"=> 2.41711e-06, "mille"=> 7.031462e-07, "mills"=> 9.690029999999999e-06, "milor"=> 6.758336e-08, "milos"=> 1.625444e-07, "milpa"=> 6.625314e-08, "milts"=> 2.6253519999999996e-09, "milty"=> 3.2108e-08, "miltz"=> 9.705358e-09, "mimed"=> 2.605264e-07, "mimeo"=> 3.072784e-07, "mimer"=> 1.5596535999999998e-08, "mimes"=> 1.890944e-07, "mimic"=> 3.6983480000000003e-06, "mimsy"=> 4.6914299999999995e-08, "minae"=> 4.105818e-08, "minar"=> 7.274627999999999e-08, "minas"=> 6.257456e-07, "mince"=> 8.868963999999999e-07, "mincy"=> 1.15748e-08, "minds"=> 2.690836e-05, "mined"=> 1.579636e-06, "miner"=> 2.54743e-06, "mines"=> 8.73529e-06, "minge"=> 2.0598500000000004e-08, "mings"=> 2.075896e-08, "mingy"=> 9.601294e-09, "minim"=> 1.633376e-07, "minis"=> 1.161506e-07, "minke"=> 9.383094e-08, "minks"=> 9.66741e-08, "minny"=> 1.1147114000000001e-07, "minor"=> 2.893282e-05, "minos"=> 3.4694540000000006e-07, "mints"=> 4.6776340000000005e-07, "minty"=> 4.2630160000000003e-07, "minus"=> 5.3614940000000005e-06, "mired"=> 8.069329999999999e-07, "mires"=> 8.064e-08, "mirex"=> 2.985702e-08, "mirid"=> 9.133378e-09, "mirin"=> 1.285004e-07, "mirks"=> 9.000504000000001e-10, "mirky"=> 2.69654e-09, "mirly"=> 1.2569552e-10, "miros"=> 8.538102000000001e-09, "mirth"=> 2.05201e-06, "mirvs"=> 2.4358799999999996e-08, "mirza"=> 7.13092e-07, "misch"=> 6.551682e-08, "misdo"=> 2.464765e-09, "miser"=> 6.115956e-07, "mises"=> 7.657604000000001e-07, "misgo"=> 3.3158423999999997e-10, "misos"=> 7.580182e-09, "missa"=> 2.200408e-07, "missy"=> 1.91137e-06, "mists"=> 1.405652e-06, "misty"=> 3.107304e-06, "mitch"=> 4.105616e-06, "miter"=> 1.6963659999999998e-07, "mites"=> 1.0248924e-06, "mitis"=> 5.5932240000000005e-08, "mitre"=> 3.966914e-07, "mitts"=> 2.783876e-07, "mixed"=> 4.088312e-05, "mixen"=> 1.6020554e-08, "mixer"=> 2.280866e-06, "mixes"=> 1.835836e-06, "mixte"=> 3.323158e-08, "mixup"=> 3.078816e-08, "mizen"=> 7.38629e-08, "mizzy"=> 2.0017394e-08, "mneme"=> 1.578735e-08, "moans"=> 2.198616e-06, "moats"=> 1.779142e-07, "mobby"=> 1.1493536e-09, "mobes"=> 1.0033425999999998e-09, "mobey"=> 1.5140787999999998e-09, "mobie"=> 2.5780804e-09, "moble"=> 2.248684e-09, "mocha"=> 5.072758e-07, "mochi"=> 8.70669e-08, "mochs"=> 1.3105689999999998e-10, "mochy"=> 2.080648e-10, "mocks"=> 5.862164e-07, "modal"=> 4.5121739999999995e-06, "model"=> 0.0002257798, "modem"=> 1.1182042e-06, "moder"=> 2.6211478e-07, "modes"=> 1.98025e-05, "modge"=> 1.4031531999999998e-09, "modii"=> 1.3942483999999999e-08, "modus"=> 1.3342560000000001e-06, "moers"=> 4.9826879999999995e-08, "mofos"=> 3.7185919999999995e-09, "moggy"=> 4.384616e-08, "mogul"=> 6.071234e-07, "mohel"=> 2.270036e-08, "mohos"=> 2.701656e-09, "mohrs"=> 4.890746e-09, "mohua"=> 7.494766e-09, "mohur"=> 1.4654432000000002e-08, "moile"=> 3.6276760000000004e-09, "moils"=> 2.880088e-09, "moira"=> 1.4368920000000002e-06, "moire"=> 5.917383999999999e-08, "moist"=> 5.0797100000000005e-06, "moits"=> 1.3802588e-10, "mojos"=> 2.023102e-08, "mokes"=> 1.1236383999999999e-08, "mokis"=> 5.4123414e-09, "mokos"=> 3.989684000000001e-09, "molal"=> 5.142052e-08, "molar"=> 3.081168e-06, "molas"=> 3.805894e-08, "molds"=> 1.302066e-06, "moldy"=> 4.2032160000000003e-07, "moled"=> 2.791244e-09, "moles"=> 1.6907859999999998e-06, "molla"=> 8.059294e-08, "molls"=> 3.25031e-08, "molly"=> 1.0789200000000002e-05, "molto"=> 3.083198e-07, "molts"=> 4.8103059999999997e-08, "molys"=> 1.6022266e-10, "momes"=> 1.0080904000000001e-09, "momma"=> 2.3446679999999997e-06, "mommy"=> 4.059056e-06, "momus"=> 7.591098e-08, "monad"=> 4.021942e-07, "monal"=> 1.340184e-08, "monas"=> 4.2141640000000005e-08, "monde"=> 2.022186e-06, "mondo"=> 6.276644000000001e-07, "moner"=> 1.0316836e-08, "money"=> 0.0001813774, "mongo"=> 1.6846019999999997e-07, "mongs"=> 9.477672e-09, "monic"=> 9.124812e-08, "monie"=> 4.108164e-08, "monks"=> 6.94092e-06, "monos"=> 5.355904e-08, "monte"=> 5.884744e-06, "month"=> 6.324196e-05, "monty"=> 1.7722400000000002e-06, "moobs"=> 4.006068e-09, "mooch"=> 9.701750000000001e-08, "moods"=> 3.4155099999999996e-06, "moody"=> 3.2951779999999996e-06, "mooed"=> 3.588262e-08, "mooks"=> 1.1233237999999999e-08, "moola"=> 2.728712e-08, "mooli"=> 1.726798e-08, "mools"=> 3.0767320000000002e-09, "mooly"=> 3.038746e-09, "moong"=> 4.0171320000000006e-08, "moons"=> 2.154184e-06, "moony"=> 9.165280000000001e-08, "moops"=> 2.823646e-09, "moors"=> 1.9608260000000002e-06, "moory"=> 4.1311160000000005e-09, "moose"=> 2.14218e-06, "moots"=> 3.43841e-08, "moove"=> 6.884039999999999e-09, "moped"=> 2.98194e-07, "moper"=> 1.8154760000000002e-09, "mopes"=> 4.0331359999999995e-08, "mopey"=> 4.50553e-08, "moppy"=> 7.600632000000002e-09, "mopsy"=> 2.836094e-08, "mopus"=> 8.032534000000002e-09, "morae"=> 2.5712820000000002e-08, "moral"=> 7.87428e-05, "moras"=> 6.479048000000001e-08, "morat"=> 3.699406e-08, "moray"=> 4.845202e-07, "morel"=> 9.092972000000001e-07, "mores"=> 1.2882420000000001e-06, "moria"=> 1.1150492e-07, "morne"=> 1.0192148e-07, "morns"=> 1.4391039999999999e-08, "moron"=> 7.329484e-07, "morph"=> 7.391076e-07, "morra"=> 1.0443672000000001e-07, "morro"=> 2.112404e-07, "morse"=> 2.532724e-06, "morts"=> 2.2034879999999997e-07, "mosed"=> 1.1244657999999998e-09, "moses"=> 2.295472e-05, "mosey"=> 1.0487743999999999e-07, "mosks"=> 1.2235746e-09, "mosso"=> 5.312906e-08, "mossy"=> 8.798876e-07, "moste"=> 9.784774e-08, "mosts"=> 4.385822e-09, "moted"=> 1.5122340000000004e-08, "motel"=> 3.467844e-06, "moten"=> 1.1144508000000002e-07, "motes"=> 3.9284520000000003e-07, "motet"=> 1.9835499999999998e-07, "motey"=> 5.5828768e-09, "moths"=> 1.33791e-06, "mothy"=> 7.3022479999999995e-09, "motif"=> 4.473966e-06, "motis"=> 6.993072000000001e-09, "motor"=> 3.2188179999999996e-05, "motte"=> 3.6770319999999995e-07, "motto"=> 2.922292e-06, "motts"=> 1.785736e-08, "motty"=> 1.5030254e-08, "motus"=> 1.0760018000000002e-07, "motza"=> 7.360126e-09, "mouch"=> 5.747864e-09, "moues"=> 3.689548e-09, "mould"=> 3.0424920000000003e-06, "mouls"=> 1.6117954000000001e-09, "moult"=> 2.6231339999999997e-07, "mound"=> 4.893352e-06, "mount"=> 1.9633919999999996e-05, "moups"=> 1.1015416e-10, "mourn"=> 2.559808e-06, "mouse"=> 1.4586620000000002e-05, "moust"=> 6.408836e-09, "mousy"=> 2.5307380000000005e-07, "mouth"=> 0.00011839139999999999, "moved"=> 0.0001131278, "mover"=> 1.345772e-06, "moves"=> 2.750236e-05, "movie"=> 2.525518e-05, "mowas"=> 4.6298062e-10, "mowed"=> 4.912136e-07, "mower"=> 7.017116e-07, "mowra"=> 3.4910540000000003e-10, "moxas"=> 2.435483e-09, "moxie"=> 2.0430499999999998e-07, "moyas"=> 5.985623999999999e-10, "moyle"=> 1.2202059999999998e-07, "moyls"=> 6.713988000000001e-10, "mozed"=> 8.406384000000001e-11, "mozes"=> 1.4875274e-07, "mozos"=> 1.867878e-08, "mpret"=> 6.651736e-10, "mucho"=> 5.902075999999999e-07, "mucic"=> 1.0790868000000001e-08, "mucid"=> 1.0534826e-09, "mucin"=> 4.320912e-07, "mucks"=> 2.840928e-08, "mucky"=> 1.611552e-07, "mucor"=> 1.1352698000000001e-07, "mucro"=> 7.925326000000001e-09, "mucus"=> 1.91744e-06, "muddy"=> 4.566792000000001e-06, "mudge"=> 2.313302e-07, "mudir"=> 2.4648459999999998e-08, "mudra"=> 2.23188e-07, "muffs"=> 8.711635999999999e-08, "mufti"=> 5.393240000000001e-07, "mugga"=> 4.401435999999999e-09, "muggs"=> 2.7670079999999998e-08, "muggy"=> 2.955084e-07, "muhly"=> 2.8846839999999995e-08, "muids"=> 7.060864e-09, "muils"=> 1.6187675999999998e-10, "muirs"=> 1.709824e-08, "muist"=> 3.87205e-10, "mujik"=> 1.2050832000000001e-08, "mulch"=> 7.15707e-07, "mulct"=> 2.438622e-08, "muled"=> 2.5327080000000002e-09, "mules"=> 2.374704e-06, "muley"=> 8.646944000000001e-08, "mulga"=> 4.858918e-08, "mulie"=> 1.7428162e-09, "mulla"=> 1.735916e-07, "mulls"=> 5.1702780000000006e-08, "mulse"=> 4.896638e-10, "mulsh"=> 1.164387e-10, "mumms"=> 1.1207092e-09, "mummy"=> 3.6918080000000006e-06, "mumps"=> 7.46224e-07, "mumsy"=> 2.7062479999999995e-08, "mumus"=> 9.366362e-10, "munch"=> 7.101918000000001e-07, "munga"=> 1.0462662e-08, "munge"=> 5.764341999999999e-09, "mungo"=> 3.915172e-07, "mungs"=> 9.306344000000001e-10, "munis"=> 4.543582e-08, "munts"=> 8.15547e-09, "muntu"=> 2.4203000000000002e-08, "muons"=> 1.4410720000000002e-07, "mural"=> 1.7756819999999997e-06, "muras"=> 7.16106e-09, "mured"=> 4.733732e-09, "mures"=> 1.7992380000000002e-08, "murex"=> 5.524098e-08, "murid"=> 5.109456e-08, "murks"=> 2.63628e-09, "murky"=> 1.859152e-06, "murls"=> 2.4457512e-10, "murly"=> 3.520474e-10, "murra"=> 4.402276e-08, "murre"=> 3.3124180000000004e-08, "murri"=> 3.1652819999999995e-08, "murrs"=> 2.0429144e-09, "murry"=> 2.68871e-07, "murti"=> 6.484018e-08, "murva"=> 8.59189e-10, "musar"=> 3.4657520000000004e-08, "musca"=> 1.0387988e-07, "mused"=> 3.2120760000000003e-06, "muser"=> 1.3140173999999998e-08, "muses"=> 1.404414e-06, "muset"=> 4.266076e-09, "musha"=> 5.085408e-08, "mushy"=> 5.734076e-07, "music"=> 0.000119424, "musit"=> 7.4159146e-10, "musks"=> 2.5618900000000004e-08, "musky"=> 5.7016e-07, "musos"=> 1.0061525999999999e-08, "musse"=> 1.204801e-08, "mussy"=> 2.463856e-08, "musth"=> 2.744112e-08, "musts"=> 8.741021999999999e-08, "musty"=> 1.194424e-06, "mutch"=> 8.77712e-08, "muted"=> 2.5731860000000002e-06, "muter"=> 2.8049240000000002e-08, "mutes"=> 2.2168600000000002e-07, "mutha"=> 3.74073e-08, "mutis"=> 2.3426860000000004e-08, "muton"=> 4.170736e-09, "mutts"=> 1.0039038e-07, "muxed"=> 1.0154478e-09, "muxes"=> 1.7103742e-08, "muzak"=> 9.171636000000001e-08, "muzzy"=> 7.399948e-08, "mvule"=> 1.2566428e-09, "myall"=> 4.819193999999999e-08, "mylar"=> 1.305934e-07, "mynah"=> 4.32605e-08, "mynas"=> 1.1002198000000002e-08, "myoid"=> 2.877644e-08, "myoma"=> 5.322592e-08, "myope"=> 6.720618e-09, "myops"=> 2.6631379999999997e-09, "myopy"=> 2.4808659999999997e-10, "myrrh"=> 6.167824000000001e-07, "mysid"=> 1.1476056e-08, "mythi"=> 2.0402742e-09, "myths"=> 7.664996e-06, "mythy"=> 9.37728e-10, "myxos"=> 2.4207354e-10, "mzees"=> 1.24260642e-09, "naams"=> 1.415992002e-08, "naans"=> 9.059624e-09, "nabes"=> 4.201108e-09, "nabis"=> 5.880274000000001e-08, "nabks"=> 0.0, "nabla"=> 2.6051e-08, "nabob"=> 1.8768750000000002e-07, "nache"=> 2.617118e-09, "nacho"=> 2.1414279999999998e-07, "nacre"=> 8.392104e-08, "nadas"=> 1.1533288e-08, "nadir"=> 8.264356e-07, "naeve"=> 1.3512018e-08, "naevi"=> 6.522379999999999e-08, "naffs"=> 7.375847999999999e-10, "nagas"=> 2.110124e-07, "naggy"=> 6.77458e-09, "nagor"=> 3.2957260000000005e-09, "nahal"=> 7.36509e-08, "naiad"=> 1.193972e-07, "naifs"=> 5.4524200000000005e-09, "naiks"=> 3.476074e-09, "nails"=> 8.742436e-06, "naira"=> 1.563248e-07, "nairu"=> 4.723484e-08, "naive"=> 3.747224e-06, "naked"=> 2.2634839999999998e-05, "naker"=> 3.594352e-09, "nakfa"=> 6.816528e-09, "nalas"=> 9.63427e-09, "naled"=> 6.728344e-09, "nalla"=> 2.628224e-08, "named"=> 5.23108e-05, "namer"=> 3.557128e-08, "names"=> 6.114654e-05, "namma"=> 1.831742e-08, "namus"=> 2.7592919999999995e-08, "nanas"=> 2.043872e-08, "nance"=> 6.268852e-07, "nancy"=> 1.201142e-05, "nandu"=> 3.077294e-08, "nanna"=> 3.551422e-07, "nanny"=> 2.8351919999999998e-06, "nanos"=> 1.1179763999999998e-07, "nanua"=> 9.145946000000001e-10, "napas"=> 1.61306e-08, "naped"=> 2.7478539999999996e-08, "napes"=> 1.8782459999999997e-08, "napoo"=> 6.286926e-09, "nappa"=> 3.3110099999999996e-08, "nappe"=> 6.974094e-08, "nappy"=> 4.066484e-07, "naras"=> 8.487691999999999e-09, "narco"=> 2.1853339999999998e-07, "narcs"=> 2.561922e-08, "nards"=> 3.3590619999999996e-09, "nares"=> 2.725834e-07, "naric"=> 8.089388e-09, "naris"=> 4.849534000000001e-08, "narks"=> 6.5943720000000005e-09, "narky"=> 4.0455866000000005e-08, "narre"=> 5.1351459999999995e-09, "nasal"=> 6.717342000000001e-06, "nashi"=> 7.112222e-08, "nasty"=> 6.77708e-06, "natal"=> 2.152084e-06, "natch"=> 5.2902040000000006e-08, "nates"=> 5.489444000000001e-08, "natis"=> 1.0004728e-08, "natty"=> 4.00965e-07, "nauch"=> 4.41196e-09, "naunt"=> 1.420679e-09, "naval"=> 1.436504e-05, "navar"=> 1.5881732e-08, "navel"=> 1.277632e-06, "naves"=> 3.01158e-07, "navew"=> 4.663326e-10, "navvy"=> 1.1268528e-07, "nawab"=> 3.3664819999999997e-07, "nazes"=> 1.324797e-10, "nazir"=> 1.7652980000000002e-07, "nazis"=> 4.771734e-06, "nduja"=> 1.6389007999999998e-08, "neafe"=> 2.0075681999999999e-10, "neals"=> 1.1992746000000002e-08, "neaps"=> 1.6785499999999998e-08, "nears"=> 3.457622e-07, "neath"=> 3.261638e-07, "neats"=> 9.196961999999999e-09, "nebek"=> 1.2995312000000001e-09, "nebel"=> 8.775002e-08, "necks"=> 2.837706e-06, "neddy"=> 1.0863407999999999e-07, "needs"=> 0.0001385542, "needy"=> 3.5299500000000004e-06, "neeld"=> 1.6760302000000003e-08, "neele"=> 1.8569259999999996e-08, "neemb"=> 9.155829999999999e-11, "neems"=> 3.9705640000000004e-09, "neeps"=> 2.488282e-08, "neese"=> 4.372208e-08, "neeze"=> 1.7171888e-09, "negro"=> 1.0794185999999999e-05, "negus"=> 1.625486e-07, "neifs"=> 5.835806e-10, "neigh"=> 3.15332e-07, "neist"=> 1.926494e-08, "neive"=> 6.087113999999999e-09, "nelis"=> 4.023518e-08, "nelly"=> 1.2094896e-06, "nemas"=> 2.0947879999999997e-09, "nemns"=> 3.473918e-11, "nempt"=> 5.688854e-10, "nenes"=> 6.0506019999999996e-09, "neons"=> 1.341266e-08, "neper"=> 7.3851399999999996e-09, "nepit"=> 2.7901902e-10, "neral"=> 2.3620939999999998e-08, "nerds"=> 3.29153e-07, "nerdy"=> 3.2784740000000006e-07, "nerka"=> 1.3746739999999999e-08, "nerks"=> 5.2227696e-10, "nerol"=> 1.6739852000000002e-08, "nerts"=> 2.3583999999999996e-09, "nertz"=> 1.8392996e-09, "nerve"=> 2.7405920000000003e-05, "nervy"=> 1.6218359999999997e-07, "nests"=> 2.730226e-06, "netes"=> 3.3716955999999997e-09, "netop"=> 4.808801999999999e-09, "netts"=> 2.39414e-09, "netty"=> 7.892461999999999e-08, "neuks"=> 7.85925e-10, "neume"=> 1.034071e-08, "neums"=> 1.5260445999999998e-09, "nevel"=> 2.033692e-08, "never"=> 0.00046983179999999993, "neves"=> 2.983706e-07, "nevus"=> 4.87866e-07, "newbs"=> 4.403519999999999e-09, "newed"=> 6.6821439999999995e-09, "newel"=> 1.1789762000000001e-07, "newer"=> 5.585931999999999e-06, "newie"=> 1.7881118e-09, "newly"=> 2.301662e-05, "newsy"=> 4.3025059999999996e-08, "newts"=> 1.3477120000000002e-07, "nexts"=> 4.728542e-09, "nexus"=> 3.5047540000000003e-06, "ngaio"=> 3.026692e-08, "ngana"=> 4.363852e-09, "ngati"=> 9.213514e-08, "ngoma"=> 5.481242e-08, "ngwee"=> 2.0567560000000002e-09, "nicad"=> 1.0218372e-08, "nicer"=> 1.8015199999999997e-06, "niche"=> 4.919476e-06, "nicht"=> 7.198515999999999e-06, "nicks"=> 3.0575299999999997e-07, "nicol"=> 5.047172e-07, "nidal"=> 1.0568478000000002e-07, "nided"=> 4.421376e-11, "nides"=> 6.5286139999999995e-09, "nidor"=> 2.5880268000000004e-09, "nidus"=> 1.505494e-07, "niece"=> 5.659456e-06, "niefs"=> 4.617968e-10, "nieve"=> 8.389590000000001e-08, "nifes"=> 1.708918e-09, "niffs"=> 3.63839e-10, "niffy"=> 3.0323639999999997e-09, "nifty"=> 3.0565940000000005e-07, "niger"=> 2.41159e-06, "nighs"=> 8.235378000000001e-10, "night"=> 0.0002603856, "nihil"=> 4.3300900000000006e-07, "nikab"=> 8.245331999999998e-10, "nikah"=> 4.5599e-08, "nikau"=> 1.3850592000000002e-08, "nills"=> 7.005867999999999e-09, "nimbi"=> 7.108082000000001e-09, "nimbs"=> 4.780258e-10, "nimps"=> 1.2160416000000002e-09, "niner"=> 9.917684000000002e-08, "nines"=> 3.595536e-07, "ninja"=> 8.948229999999999e-07, "ninny"=> 1.6861899999999998e-07, "ninon"=> 6.791969999999999e-08, "ninth"=> 8.289214e-06, "nipas"=> 1.1105497999999999e-08, "nippy"=> 9.519857999999999e-08, "niqab"=> 1.39031e-07, "nirls"=> 8.363358e-11, "nirly"=> 3.5090439999999995e-10, "nisei"=> 2.7379040000000004e-07, "nisse"=> 4.66846e-08, "nisus"=> 7.816092000000001e-08, "niter"=> 4.03585e-08, "nites"=> 1.1286557999999998e-08, "nitid"=> 2.1673284e-09, "niton"=> 1.1522543999999999e-08, "nitre"=> 9.976002000000001e-08, "nitro"=> 6.093365999999999e-07, "nitry"=> 1.3781924000000003e-09, "nitty"=> 2.380074e-07, "nival"=> 1.591197e-08, "nixed"=> 9.46801e-08, "nixer"=> 2.8974844000000004e-09, "nixes"=> 1.4469958e-08, "nixie"=> 1.0213918e-07, "nizam"=> 3.250714e-07, "nkosi"=> 8.896624e-08, "noahs"=> 2.3905130000000003e-08, "nobby"=> 1.985456e-07, "noble"=> 2.1876179999999998e-05, "nobly"=> 8.836094e-07, "nocks"=> 1.7302364000000002e-08, "nodal"=> 2.0325739999999995e-06, "noddy"=> 1.2921234e-07, "nodes"=> 1.856478e-05, "nodus"=> 2.301028e-08, "noels"=> 2.7158779999999998e-08, "noggs"=> 3.9839566e-08, "nohow"=> 1.1631908e-07, "noils"=> 2.667196e-09, "noily"=> 4.6887128e-10, "noint"=> 5.195795999999999e-09, "noirs"=> 2.4103260000000004e-07, "noise"=> 4.466944e-05, "noisy"=> 5.850514e-06, "noles"=> 1.535012e-08, "nolls"=> 1.8601192000000001e-09, "nolos"=> 1.5307724e-10, "nomad"=> 7.42969e-07, "nomas"=> 3.058938e-08, "nomen"=> 3.059724e-07, "nomes"=> 1.0446661999999999e-07, "nomic"=> 2.0282860000000003e-07, "nomoi"=> 4.61669e-08, "nomos"=> 6.438258e-07, "nonas"=> 9.588335999999999e-09, "nonce"=> 3.2864259999999997e-07, "nones"=> 2.0841039999999997e-07, "nonet"=> 2.3018720000000003e-08, "nongs"=> 8.49868e-10, "nonis"=> 8.933246e-09, "nonny"=> 7.047221999999999e-08, "nonyl"=> 1.8421820000000003e-08, "noobs"=> 3.761134e-08, "nooit"=> 2.544494e-07, "nooks"=> 5.301522e-07, "nooky"=> 1.1360314000000001e-08, "noons"=> 3.48012e-08, "noops"=> 2.3670236000000003e-09, "noose"=> 1.238534e-06, "nopal"=> 4.659296e-08, "noria"=> 3.40507e-08, "noris"=> 4.4836719999999997e-08, "norks"=> 2.599074e-09, "norma"=> 1.697114e-06, "norms"=> 2.427588e-05, "north"=> 0.0001298898, "nosed"=> 1.356692e-06, "noser"=> 1.335376e-08, "noses"=> 2.938642e-06, "nosey"=> 2.8319880000000004e-07, "notal"=> 5.764468e-09, "notch"=> 3.768332e-06, "noted"=> 6.841880000000001e-05, "noter"=> 2.1284840000000004e-08, "notes"=> 7.86177e-05, "notum"=> 2.69894e-08, "nould"=> 8.716478e-10, "noule"=> 2.355495e-10, "nouls"=> 5.091527999999999e-10, "nouns"=> 4.465028e-06, "nouny"=> 2.370482e-09, "noups"=> 4.4922520000000006e-11, "novae"=> 1.1663219999999999e-07, "novas"=> 1.047251e-07, "novel"=> 5.154866000000001e-05, "novum"=> 3.314432e-07, "noway"=> 3.197006e-08, "nowed"=> 8.168665999999999e-10, "nowls"=> 3.3578826e-10, "nowts"=> 7.392325999999999e-10, "nowty"=> 3.3588340000000003e-10, "noxal"=> 4.501214e-09, "noxes"=> 1.4914606000000001e-09, "noyau"=> 2.07878e-08, "noyed"=> 2.214484e-09, "noyes"=> 5.324632e-07, "nubby"=> 4.7307559999999995e-08, "nubia"=> 3.703974e-07, "nucha"=> 3.993998e-09, "nuddy"=> 3.437412e-09, "nuder"=> 4.317471999999999e-09, "nudes"=> 2.667734e-07, "nudge"=> 1.8194560000000002e-06, "nudie"=> 3.952014e-08, "nudzh"=> 1.4392722e-10, "nuffs"=> 6.591171999999999e-10, "nugae"=> 1.4009939999999998e-08, "nuked"=> 4.95597e-08, "nukes"=> 1.911538e-07, "nulla"=> 9.995526e-07, "nulls"=> 1.183814e-07, "numbs"=> 6.951156e-08, "numen"=> 1.3415880000000002e-07, "nummy"=> 4.505146e-09, "nunny"=> 5.206336e-09, "nurds"=> 3.0151764e-10, "nurdy"=> 7.040558e-11, "nurls"=> 4.072635e-10, "nurrs"=> 5.785664e-11, "nurse"=> 2.986808e-05, "nutso"=> 1.9746439999999998e-08, "nutsy"=> 7.51611e-09, "nutty"=> 6.938174e-07, "nyaff"=> 1.1056072e-09, "nyala"=> 3.4069759999999996e-08, "nying"=> 1.260617e-08, "nylon"=> 1.8047460000000003e-06, "nymph"=> 1.388424e-06, "nyssa"=> 4.3694060000000005e-07, "oaked"=> 1.784058e-08, "oaken"=> 4.17275e-07, "oaker"=> 3.019748e-09, "oakum"=> 1.0791358000000001e-07, "oared"=> 8.275623999999999e-08, "oases"=> 3.459664e-07, "oasis"=> 2.195334e-06, "oasts"=> 4.754582e-09, "oaten"=> 5.490832e-08, "oater"=> 4.75045e-09, "oaths"=> 2.0643239999999997e-06, "oaves"=> 5.253176e-10, "obang"=> 2.4115824e-09, "obeah"=> 1.705577e-07, "obeli"=> 2.0788400000000002e-09, "obese"=> 3.59957e-06, "obeys"=> 1.0131524e-06, "obias"=> 6.893422000000001e-09, "obied"=> 2.343832e-09, "obiit"=> 2.0560199999999998e-08, "obits"=> 4.3761e-08, "objet"=> 2.9833199999999995e-07, "oboes"=> 8.979082e-08, "obole"=> 3.753844e-09, "oboli"=> 5.016546e-09, "obols"=> 3.6793619999999996e-08, "occam"=> 2.286222e-07, "occur"=> 6.082456e-05, "ocean"=> 3.273e-05, "ocher"=> 1.265226e-07, "oches"=> 2.429964e-09, "ochre"=> 6.25487e-07, "ochry"=> 4.3203820000000004e-10, "ocker"=> 3.7664100000000004e-08, "ocrea"=> 6.2508659999999995e-09, "octad"=> 5.6019639999999996e-09, "octal"=> 1.375002e-07, "octan"=> 1.0645828e-08, "octas"=> 1.5790798e-09, "octet"=> 3.278916e-07, "octyl"=> 9.599716000000001e-08, "oculi"=> 1.8319540000000002e-07, "odahs"=> 8.73975e-11, "odals"=> 3.9593199999999996e-10, "odder"=> 1.6255040000000003e-07, "oddly"=> 4.796204e-06, "odeon"=> 1.725502e-07, "odeum"=> 1.1409482e-08, "odism"=> 1.2986546e-10, "odist"=> 9.208184000000001e-09, "odium"=> 2.3106240000000003e-07, "odors"=> 1.301368e-06, "odour"=> 1.7990779999999999e-06, "odyle"=> 2.859056e-09, "odyls"=> 0.0, "ofays"=> 2.343572e-09, "offal"=> 3.51852e-07, "offed"=> 5.9569400000000004e-08, "offer"=> 8.4951e-05, "offie"=> 1.2245437999999999e-08, "oflag"=> 2.436006e-08, "often"=> 0.000318198, "ofter"=> 1.151668e-08, "ogams"=> 1.731963e-09, "ogeed"=> 2.5068862000000005e-10, "ogees"=> 2.077176e-09, "oggin"=> 1.6686806e-09, "ogham"=> 6.111918000000001e-08, "ogive"=> 6.724802e-08, "ogled"=> 1.56177e-07, "ogler"=> 5.4626979999999995e-09, "ogles"=> 3.802444e-08, "ogmic"=> 3.5693204e-10, "ogres"=> 2.980896e-07, "ohias"=> 7.943790999999999e-10, "ohing"=> 1.337772e-09, "ohmic"=> 4.413572e-07, "ohone"=> 2.448318e-09, "oidia"=> 3.0581182e-09, "oiled"=> 1.2047860000000001e-06, "oiler"=> 1.0617669999999998e-07, "oinks"=> 5.9977440000000005e-09, "oints"=> 2.5817840000000002e-08, "ojime"=> 8.826421999999999e-10, "okapi"=> 4.358124e-08, "okays"=> 1.82243e-08, "okehs"=> 1.45804e-10, "okras"=> 5.300552e-09, "oktas"=> 2.35481e-09, "olden"=> 8.317202000000001e-07, "older"=> 7.048674e-05, "oldie"=> 6.778512e-08, "oleic"=> 4.191254e-07, "olein"=> 2.5967779999999997e-08, "olent"=> 3.740556e-09, "oleos"=> 1.9849820000000005e-09, "oleum"=> 5.741464e-08, "olios"=> 2.5280119999999996e-09, "olive"=> 1.582624e-05, "ollas"=> 2.150638e-08, "ollav"=> 2.002796e-09, "oller"=> 9.310342e-08, "ollie"=> 1.852534e-06, "ology"=> 1.2207628e-07, "olpae"=> 2.747946e-09, "olpes"=> 1.4735266e-10, "omasa"=> 4.98492e-09, "omber"=> 4.926426000000001e-09, "ombre"=> 1.7465000000000002e-07, "ombus"=> 3.427348e-10, "omega"=> 3.5414939999999993e-06, "omens"=> 7.610148e-07, "omers"=> 3.135222e-08, "omits"=> 1.0365344e-06, "omlah"=> 4.4520585999999997e-10, "omovs"=> 0.0, "omrah"=> 7.597728e-09, "oncer"=> 2.235548e-09, "onces"=> 7.656184e-09, "oncet"=> 2.7724579999999997e-08, "oncus"=> 3.0811999999999997e-10, "onely"=> 5.634636000000001e-07, "oners"=> 1.173823e-08, "onery"=> 7.392516000000001e-09, "onion"=> 8.780275999999999e-06, "onium"=> 1.69328e-08, "onkus"=> 2.0803952000000002e-10, "onlay"=> 8.695974e-08, "onned"=> 3.8960056e-10, "onset"=> 1.5109339999999999e-05, "ontic"=> 2.907526e-07, "oobit"=> 4.4796381999999993e-10, "oohed"=> 7.348398e-08, "oomph"=> 1.673814e-07, "oonts"=> 9.452378e-10, "ooped"=> 7.535025999999999e-10, "oorie"=> 3.2602600000000003e-10, "ooses"=> 4.3164900000000005e-10, "ootid"=> 2.307722e-09, "oozed"=> 8.500759999999999e-07, "oozes"=> 2.1615980000000001e-07, "opahs"=> 1.4624736000000001e-09, "opals"=> 1.4276300000000002e-07, "opens"=> 1.6933800000000004e-05, "opepe"=> 1.6193570000000002e-09, "opera"=> 1.2324200000000003e-05, "opine"=> 1.991092e-07, "oping"=> 5.044878e-08, "opium"=> 3.5690459999999996e-06, "oppos"=> 1.1225878e-08, "opsin"=> 1.015625e-07, "opted"=> 3.906028e-06, "opter"=> 3.579358e-09, "optic"=> 5.0226200000000005e-06, "orach"=> 1.6915602000000003e-08, "oracy"=> 4.7509259999999996e-08, "orals"=> 3.209446e-08, "orang"=> 4.495462e-07, "orant"=> 1.1761404e-08, "orate"=> 5.513046e-08, "orbed"=> 4.8426600000000006e-08, "orbit"=> 7.737666e-06, "orcas"=> 1.980194e-07, "orcin"=> 2.0968352e-09, "order"=> 0.00030720620000000004, "ordos"=> 6.17266e-08, "oread"=> 1.55142e-08, "orfes"=> 2.1499820000000003e-10, "organ"=> 1.64243e-05, "orgia"=> 7.443383999999999e-09, "orgic"=> 1.79757e-10, "orgue"=> 2.57739e-08, "oribi"=> 1.430926e-08, "oriel"=> 3.898128e-07, "orixa"=> 5.048654e-09, "orles"=> 1.8498437999999999e-09, "orlon"=> 2.380896e-08, "orlop"=> 3.009208e-08, "ormer"=> 1.818116e-08, "ornis"=> 1.4785822e-08, "orpin"=> 6.6936840000000004e-09, "orris"=> 8.974691999999999e-08, "ortho"=> 6.31145e-07, "orval"=> 6.015791999999999e-08, "orzos"=> 1.7806792e-10, "oscar"=> 6.9473e-06, "oshac"=> 1.2070638000000002e-10, "osier"=> 8.59745e-08, "osmic"=> 1.3616140000000003e-08, "osmol"=> 1.1496445999999999e-08, "ossia"=> 2.7330560000000004e-08, "ostia"=> 4.013904e-07, "otaku"=> 1.7126392000000002e-07, "otary"=> 1.9770974e-09, "other"=> 0.0012753340000000001, "ottar"=> 6.725306e-08, "otter"=> 1.411804e-06, "ottos"=> 1.3527379999999998e-08, "oubit"=> 6.045934e-10, "oucht"=> 2.3286232e-09, "ouens"=> 1.632076e-08, "ought"=> 3.826963999999999e-05, "ouija"=> 2.326276e-07, "oulks"=> 6.899324000000001e-11, "oumas"=> 1.6793888000000002e-09, "ounce"=> 5.06985e-06, "oundy"=> 3.18417e-10, "oupas"=> 4.536068e-10, "ouped"=> 3.26366e-10, "ouphe"=> 1.554769e-09, "ouphs"=> 6.475516e-10, "ourie"=> 9.335462e-10, "ousel"=> 1.385034e-08, "ousts"=> 2.5489339999999997e-08, "outby"=> 1.1744476e-08, "outdo"=> 4.3765700000000004e-07, "outed"=> 2.086568e-07, "outer"=> 2.27653e-05, "outgo"=> 4.262226e-08, "outre"=> 2.026024e-07, "outro"=> 2.1383844e-07, "outta"=> 1.208773e-06, "ouzel"=> 3.491526e-08, "ouzos"=> 9.338096e-10, "ovals"=> 2.774458e-07, "ovary"=> 1.9248839999999995e-06, "ovate"=> 3.449494e-07, "ovels"=> 7.924934e-09, "ovens"=> 1.0422450000000001e-06, "overs"=> 7.587130000000001e-07, "overt"=> 4.984375999999999e-06, "ovine"=> 1.7328340000000003e-07, "ovist"=> 1.4611198e-09, "ovoid"=> 4.2858560000000003e-07, "ovoli"=> 1.2309804000000002e-09, "ovolo"=> 1.4397619999999997e-08, "ovule"=> 1.691038e-07, "owche"=> 2.9147272e-10, "owies"=> 3.58916e-09, "owing"=> 1.0806098e-05, "owled"=> 5.533082e-09, "owler"=> 1.3986763999999999e-08, "owlet"=> 4.708558e-08, "owned"=> 2.8682299999999995e-05, "owner"=> 3.588356e-05, "owres"=> 7.664178e-10, "owrie"=> 1.0043528e-09, "owsen"=> 2.874674e-09, "oxbow"=> 3.767556e-07, "oxers"=> 1.6395136e-09, "oxeye"=> 1.337958e-08, "oxide"=> 1.1343413999999999e-05, "oxids"=> 4.0214960000000003e-10, "oxies"=> 1.054648e-09, "oxime"=> 1.2452560000000002e-07, "oxims"=> 1.6395284000000002e-10, "oxlip"=> 9.561039999999998e-09, "oxter"=> 8.765782e-09, "oyers"=> 9.046186000000002e-10, "ozeki"=> 4.116738e-08, "ozone"=> 3.7657219999999996e-06, "ozzie"=> 3.237742e-07, "paals"=> 4.770279e-10, "paans"=> 5.6027480000000005e-09, "pacas"=> 1.188389e-08, "paced"=> 5.098298e-06, "pacer"=> 1.742384e-07, "paces"=> 3.8163120000000005e-06, "pacey"=> 1.824036e-07, "pacha"=> 1.546603e-07, "packs"=> 3.467554e-06, "pacos"=> 1.146101e-08, "pacta"=> 1.0936508e-07, "pacts"=> 4.1715e-07, "paddy"=> 2.67573e-06, "padis"=> 3.883672e-09, "padle"=> 9.936856e-10, "padma"=> 3.60845e-07, "padre"=> 2.39285e-06, "padri"=> 1.0317227999999999e-07, "paean"=> 2.56535e-07, "paedo"=> 1.823998e-08, "paeon"=> 1.7329778000000002e-08, "pagan"=> 5.632424e-06, "paged"=> 2.730668e-07, "pager"=> 3.556462e-07, "pages"=> 3.626398e-05, "pagle"=> 6.85833e-10, "pagod"=> 9.685682e-09, "pagri"=> 7.824778e-09, "paiks"=> 5.8559e-09, "pails"=> 3.90248e-07, "pains"=> 7.847448e-06, "paint"=> 1.85821e-05, "paire"=> 6.218942e-08, "pairs"=> 1.544826e-05, "paisa"=> 5.7753379999999996e-08, "paise"=> 7.020941999999999e-08, "pakka"=> 1.341967e-08, "palas"=> 4.411952e-08, "palay"=> 2.494934e-08, "palea"=> 4.784432e-08, "paled"=> 1.3991860000000001e-06, "paler"=> 1.497338e-06, "pales"=> 3.43649e-07, "palet"=> 1.3228978e-08, "palis"=> 2.2247300000000004e-08, "palki"=> 1.3228538e-08, "palla"=> 1.2085522e-07, "palls"=> 4.633776e-08, "pally"=> 6.234478e-08, "palms"=> 9.436604e-06, "palmy"=> 6.79217e-08, "palpi"=> 3.156328e-08, "palps"=> 6.395634e-08, "palsa"=> 5.788108e-09, "palsy"=> 2.333458e-06, "pampa"=> 1.772324e-07, "panax"=> 1.0631410000000001e-07, "pance"=> 8.53264e-09, "panda"=> 9.557192e-07, "pands"=> 1.7448120000000001e-09, "pandy"=> 8.836572e-08, "paned"=> 1.747922e-07, "panel"=> 2.584208e-05, "panes"=> 1.34131e-06, "panga"=> 5.584988e-08, "pangs"=> 1.1087480000000001e-06, "panic"=> 1.863406e-05, "panim"=> 2.144368e-08, "panko"=> 1.819732e-07, "panne"=> 6.707438000000001e-08, "panni"=> 2.82132e-08, "pansy"=> 6.31534e-07, "panto"=> 6.012928e-08, "pants"=> 1.492222e-05, "panty"=> 3.28287e-07, "paoli"=> 2.389044e-07, "paolo"=> 2.320586e-06, "papal"=> 3.8223059999999996e-06, "papas"=> 1.8516839999999998e-07, "papaw"=> 7.969442e-08, "paper"=> 0.0001220696, "papes"=> 3.13534e-08, "pappi"=> 7.793124e-08, "pappy"=> 4.0417379999999997e-07, "parae"=> 6.7106182e-09, "paras"=> 3.2242419999999994e-06, "parch"=> 5.487418e-08, "pardi"=> 3.9831220000000005e-08, "pards"=> 4.1695420000000005e-08, "pardy"=> 2.262718e-08, "pared"=> 4.655452e-07, "paren"=> 3.6417800000000005e-08, "pareo"=> 7.897122e-09, "parer"=> 3.100232e-08, "pares"=> 1.8205019999999998e-07, "pareu"=> 5.7311379999999995e-09, "parev"=> 1.3723766e-09, "parge"=> 5.273074000000001e-09, "pargo"=> 1.0046994e-08, "paris"=> 5.8291720000000004e-05, "parka"=> 4.833776000000001e-07, "parki"=> 1.8662984e-08, "parks"=> 1.0175154000000001e-05, "parky"=> 2.298526e-08, "parle"=> 3.05427e-07, "parly"=> 2.3122919999999996e-08, "parma"=> 7.885315999999999e-07, "parol"=> 1.9123000000000003e-07, "parps"=> 8.534269999999999e-09, "parra"=> 3.12027e-07, "parrs"=> 1.2498582e-08, "parry"=> 2.0190919999999996e-06, "parse"=> 8.604362e-07, "parti"=> 7.152798e-07, "parts"=> 9.197862000000001e-05, "party"=> 0.00017628780000000002, "parve"=> 1.4762580000000001e-08, "parvo"=> 4.605216e-08, "paseo"=> 2.680096e-07, "pases"=> 7.3056059999999995e-09, "pasha"=> 1.71792e-06, "pashm"=> 2.161452e-09, "paska"=> 1.1240076e-08, "paspy"=> 3.77015e-11, "passe"=> 5.865192e-07, "pasta"=> 5.032576e-06, "paste"=> 6.568624e-06, "pasts"=> 9.522293999999999e-07, "pasty"=> 5.403162e-07, "patch"=> 1.1440579999999999e-05, "pated"=> 6.25029e-08, "paten"=> 6.634874000000002e-08, "pater"=> 1.2936580000000001e-06, "pates"=> 5.2661920000000006e-08, "paths"=> 1.6223099999999997e-05, "patin"=> 4.9726520000000005e-08, "patio"=> 3.320716e-06, "patka"=> 1.5992814e-08, "patly"=> 2.5967779999999996e-09, "patsy"=> 1.48482e-06, "patte"=> 6.203536e-08, "patty"=> 3.3401740000000004e-06, "patus"=> 1.5884435999999999e-09, "pauas"=> 1.0087172e-10, "pauls"=> 1.8412020000000002e-07, "pause"=> 1.8683760000000003e-05, "pavan"=> 1.476884e-07, "paved"=> 4.310083999999999e-06, "paven"=> 9.325768e-09, "paver"=> 6.861002000000001e-08, "paves"=> 3.446774e-07, "pavid"=> 1.2246012e-09, "pavin"=> 5.1076200000000004e-08, "pavis"=> 4.2059660000000005e-08, "pawas"=> 1.7197334000000003e-09, "pawaw"=> 3.018904e-10, "pawed"=> 4.3615020000000004e-07, "pawer"=> 1.1323957999999998e-09, "pawks"=> 9.590566e-11, "pawky"=> 1.212484e-08, "pawls"=> 1.3875326000000002e-08, "pawns"=> 7.176781999999999e-07, "paxes"=> 1.553972e-09, "payed"=> 1.0151844e-07, "payee"=> 6.535604000000001e-07, "payer"=> 1.076341e-06, "payor"=> 2.511716e-07, "paysd"=> 1.3540072000000002e-10, "peace"=> 8.718788e-05, "peach"=> 3.4054499999999997e-06, "peage"=> 1.6147366e-09, "peags"=> 7.19233e-11, "peaks"=> 8.294124e-06, "peaky"=> 7.338202e-08, "peals"=> 3.672036e-07, "peans"=> 1.0400278e-08, "peare"=> 1.2820362e-08, "pearl"=> 1.0015582e-05, "pears"=> 1.6329640000000004e-06, "peart"=> 1.1050062e-07, "pease"=> 5.844856e-07, "peats"=> 8.662878e-08, "peaty"=> 1.273786e-07, "peavy"=> 3.79406e-08, "peaze"=> 1.085668e-09, "pebas"=> 1.5945446000000002e-09, "pecan"=> 7.841084e-07, "pechs"=> 2.5825678e-09, "pecke"=> 4.4958580000000004e-09, "pecks"=> 1.9293120000000003e-07, "pecky"=> 5.454908e-09, "pedal"=> 2.189606e-06, "pedes"=> 7.803088000000001e-08, "pedis"=> 2.1849039999999996e-07, "pedro"=> 5.042707999999999e-06, "peece"=> 5.447386e-08, "peeks"=> 4.102962e-07, "peels"=> 6.685044e-07, "peens"=> 3.4722079999999997e-09, "peeoy"=> 4.934364e-11, "peepe"=> 5.3318920000000005e-09, "peeps"=> 2.9077e-07, "peers"=> 1.33965e-05, "peery"=> 3.839226e-08, "peeve"=> 7.738294e-08, "peggy"=> 3.871502000000001e-06, "peghs"=> 3.972891999999999e-11, "peins"=> 3.42149e-09, "peise"=> 2.702074e-09, "peize"=> 9.959937999999998e-10, "pekan"=> 1.6612844e-08, "pekes"=> 3.378712e-09, "pekin"=> 1.922564e-07, "pekoe"=> 3.024258e-08, "pelas"=> 7.413232e-08, "pelau"=> 4.563478000000001e-09, "peles"=> 1.885292e-08, "pelfs"=> 1.0052536000000001e-10, "pells"=> 3.942862e-08, "pelma"=> 8.630422e-10, "pelon"=> 1.8026079999999998e-08, "pelta"=> 1.322532e-08, "pelts"=> 4.32446e-07, "penal"=> 4.414598e-06, "pence"=> 1.5524620000000002e-06, "pends"=> 1.8559e-08, "pendu"=> 1.9858659999999997e-08, "pened"=> 5.178682e-08, "penes"=> 2.6371480000000002e-08, "pengo"=> 1.538558e-08, "penie"=> 1.071075e-08, "penis"=> 4.896126e-06, "penks"=> 9.086694e-11, "penna"=> 1.998116e-07, "penne"=> 2.07646e-07, "penni"=> 3.244156e-08, "penny"=> 1.0257998e-05, "pents"=> 3.4936480000000004e-09, "peons"=> 1.632946e-07, "peony"=> 3.4584340000000003e-07, "pepla"=> 3.827637999999999e-09, "pepos"=> 9.192986000000001e-10, "peppy"=> 1.4306932e-07, "pepsi"=> 7.498654e-07, "perai"=> 3.8385919999999995e-09, "perce"=> 2.814026e-07, "perch"=> 2.54417e-06, "percs"=> 7.768036e-09, "perdu"=> 2.306308e-07, "perdy"=> 1.4642720000000001e-08, "perea"=> 1.2815040000000001e-07, "peres"=> 4.940428e-07, "peril"=> 3.952066e-06, "peris"=> 1.057133e-07, "perks"=> 1.0402744000000001e-06, "perky"=> 5.680768e-07, "perms"=> 4.019248e-08, "perns"=> 1.0514309999999999e-09, "perog"=> 9.244473999999999e-11, "perps"=> 8.874686e-08, "perry"=> 7.074313999999999e-06, "perse"=> 4.439028e-07, "perst"=> 2.9237386000000005e-09, "perts"=> 9.198332e-09, "perve"=> 1.0221254e-08, "pervo"=> 4.5094220000000004e-08, "pervs"=> 2.241864e-08, "pervy"=> 4.870926e-08, "pesky"=> 5.19979e-07, "pesos"=> 9.565264e-07, "pesto"=> 9.168344e-07, "pests"=> 2.325022e-06, "pesty"=> 7.176248e-09, "petal"=> 1.0675573999999998e-06, "petar"=> 1.1122962000000001e-07, "peter"=> 6.975812e-05, "petit"=> 2.3170979999999997e-06, "petre"=> 1.9972020000000003e-07, "petri"=> 1.2906959999999999e-06, "petti"=> 4.554044e-08, "petto"=> 5.769442000000001e-08, "petty"=> 6.786028000000001e-06, "pewee"=> 3.093134e-08, "pewit"=> 4.85107e-09, "peyse"=> 8.498334e-10, "phage"=> 1.0329689999999999e-06, "phang"=> 7.998588e-08, "phare"=> 7.75524e-08, "pharm"=> 1.7970859999999997e-06, "phase"=> 7.637994e-05, "pheer"=> 5.631849e-10, "phene"=> 2.0118540000000002e-08, "pheon"=> 1.7460528000000003e-09, "phese"=> 1.7103096000000001e-10, "phial"=> 2.5812899999999997e-07, "phish"=> 7.485769999999999e-08, "phizz"=> 1.5638662e-09, "phlox"=> 1.4875320000000002e-07, "phoca"=> 4.213686e-08, "phone"=> 9.411464e-05, "phono"=> 4.425384e-08, "phons"=> 1.0514409999999999e-08, "phony"=> 8.877268e-07, "photo"=> 2.8458819999999997e-05, "phots"=> 1.3968672000000003e-09, "phpht"=> 0.0, "phuts"=> 4.291664e-10, "phyla"=> 2.772612e-07, "phyle"=> 1.775064e-08, "piani"=> 2.7451060000000005e-08, "piano"=> 1.15262e-05, "pians"=> 2.2978380000000006e-09, "pibal"=> 1.2560783999999998e-09, "pical"=> 3.864234e-09, "picas"=> 1.8437379999999997e-08, "piccy"=> 1.4190124e-09, "picks"=> 5.191694e-06, "picky"=> 5.970651999999999e-07, "picot"=> 2.862826e-07, "picra"=> 1.8352669999999999e-09, "picul"=> 2.068946e-08, "piece"=> 6.618172e-05, "piend"=> 1.4322579999999999e-10, "piers"=> 2.063928e-06, "piert"=> 5.07146e-09, "pieta"=> 9.218749999999999e-08, "piets"=> 3.1631879999999995e-09, "piety"=> 4.73588e-06, "piezo"=> 2.2443780000000003e-07, "piggy"=> 5.727306e-07, "pight"=> 7.938189999999999e-09, "pigmy"=> 1.1019715999999999e-07, "piing"=> 5.021158e-10, "pikas"=> 4.6597000000000004e-08, "pikau"=> 5.271965e-10, "piked"=> 1.6340639999999998e-08, "piker"=> 4.6799439999999996e-08, "pikes"=> 5.28067e-07, "pikey"=> 1.7985303999999996e-08, "pikis"=> 8.122918e-09, "pikul"=> 1.357826e-08, "pilae"=> 5.449702e-09, "pilaf"=> 1.7076059999999998e-07, "pilao"=> 2.0878006000000004e-09, "pilar"=> 8.21229e-07, "pilau"=> 3.4207980000000005e-08, "pilaw"=> 2.0752540000000003e-09, "pilch"=> 5.6749260000000004e-08, "pilea"=> 8.336386e-09, "piled"=> 5.511062e-06, "pilei"=> 3.18513e-09, "piler"=> 1.3235793999999999e-08, "piles"=> 5.096398e-06, "pilis"=> 1.1932337999999999e-08, "pills"=> 5.635156e-06, "pilot"=> 2.0096700000000004e-05, "pilow"=> 5.246988000000001e-10, "pilum"=> 3.872114e-08, "pilus"=> 1.1356358000000001e-07, "pimas"=> 4.195382e-08, "pimps"=> 3.9885799999999996e-07, "pinas"=> 8.003866e-09, "pinch"=> 5.843946e-06, "pined"=> 2.898212e-07, "pines"=> 3.268782e-06, "piney"=> 2.0556020000000003e-07, "pingo"=> 2.882334e-08, "pings"=> 2.0616560000000003e-07, "pinko"=> 2.1135620000000003e-08, "pinks"=> 4.7729e-07, "pinky"=> 8.824784e-07, "pinna"=> 2.985042e-07, "pinny"=> 6.057771999999999e-08, "pinon"=> 6.001602e-08, "pinot"=> 7.571572000000001e-07, "pinta"=> 1.1334788e-07, "pinto"=> 1.5793999999999999e-06, "pints"=> 9.7577e-07, "pinup"=> 8.19493e-08, "pions"=> 1.081316e-07, "piony"=> 1.6231918e-09, "pious"=> 4.6199139999999996e-06, "pioye"=> 5.7009399999999995e-11, "pioys"=> 4.739778e-11, "pipal"=> 2.7877180000000003e-08, "pipas"=> 6.591325999999999e-09, "piped"=> 1.5354460000000002e-06, "piper"=> 4.4092719999999995e-06, "pipes"=> 6.379992e-06, "pipet"=> 7.353508e-08, "pipis"=> 6.308588e-09, "pipit"=> 7.596228e-08, "pippy"=> 2.864496e-08, "pipul"=> 2.427798e-09, "pique"=> 5.589744000000001e-07, "pirai"=> 1.0641108e-09, "pirls"=> 8.906034e-08, "pirns"=> 2.4914996e-09, "pirog"=> 1.490286e-08, "pisco"=> 1.0310191999999999e-07, "pises"=> 6.098794e-10, "pisky"=> 7.3303639999999995e-09, "pisos"=> 1.5851740000000002e-08, "pissy"=> 1.561278e-07, "piste"=> 8.78428e-08, "pitas"=> 5.505912e-08, "pitch"=> 1.5901200000000003e-05, "piths"=> 3.954759999999999e-09, "pithy"=> 4.4777040000000007e-07, "piton"=> 9.011095999999999e-08, "pitot"=> 1.59263e-07, "pitta"=> 3.06028e-07, "piums"=> 1.7081859999999998e-09, "pivot"=> 2.711766e-06, "pixel"=> 4.08482e-06, "pixes"=> 7.135892000000001e-10, "pixie"=> 8.880978e-07, "pized"=> 1.1937018e-10, "pizes"=> 1.7061104000000002e-10, "pizza"=> 7.968839999999998e-06, "plaas"=> 7.703888e-08, "place"=> 0.00041577380000000003, "plack"=> 3.241622e-08, "plage"=> 1.810554e-07, "plaid"=> 1.318204e-06, "plain"=> 3.15795e-05, "plait"=> 3.9602839999999995e-07, "plane"=> 3.9419659999999995e-05, "plank"=> 2.8246939999999997e-06, "plans"=> 5.679324e-05, "plant"=> 6.700368e-05, "plaps"=> 3.30086e-09, "plash"=> 9.251748e-08, "plasm"=> 1.4540422e-07, "plast"=> 1.118645e-06, "plate"=> 3.89231e-05, "plats"=> 1.37704e-07, "platt"=> 1.1198260000000001e-06, "platy"=> 4.650338e-08, "playa"=> 6.928954e-07, "plays"=> 3.210892000000001e-05, "plaza"=> 4.7927380000000005e-06, "plead"=> 3.3790080000000004e-06, "pleas"=> 2.084104e-06, "pleat"=> 1.287074e-07, "plebe"=> 7.959058e-08, "plebs"=> 2.447984e-07, "plena"=> 1.3018211999999998e-07, "pleon"=> 1.50681e-08, "plesh"=> 2.126136e-09, "plews"=> 1.806858e-08, "plica"=> 5.7975820000000005e-08, "plied"=> 7.623356000000001e-07, "plier"=> 2.712302e-08, "plies"=> 2.944202e-07, "plims"=> 5.280964199999999e-10, "pling"=> 3.414628e-08, "plink"=> 8.267013999999999e-08, "ploat"=> 3.290656e-10, "plods"=> 5.5148719999999996e-08, "plong"=> 3.0148300000000004e-09, "plonk"=> 8.746332e-08, "plook"=> 9.945524e-10, "plops"=> 1.46709e-07, "plots"=> 8.266088e-06, "plotz"=> 4.037568e-08, "plouk"=> 2.1233020000000002e-10, "plows"=> 3.601212e-07, "ploye"=> 9.793346e-10, "ploys"=> 1.77194e-07, "pluck"=> 2.213482e-06, "plues"=> 7.915884000000001e-10, "pluff"=> 8.018448e-09, "plugs"=> 1.186684e-06, "plumb"=> 1.0187525999999999e-06, "plume"=> 2.1018260000000002e-06, "plump"=> 3.0883779999999998e-06, "plums"=> 9.742912e-07, "plumy"=> 3.531374e-08, "plunk"=> 1.320296e-07, "pluot"=> 3.687682e-09, "plush"=> 1.622204e-06, "pluto"=> 1.846162e-06, "plyer"=> 8.159975999999999e-09, "poach"=> 2.491844e-07, "poaka"=> 2.6089187999999997e-09, "poake"=> 3.0643742e-10, "poboy"=> 3.7063400000000003e-09, "pocks"=> 2.3939999999999998e-08, "pocky"=> 1.59551e-08, "podal"=> 7.988732e-09, "poddy"=> 1.596662e-08, "podex"=> 4.2677139999999996e-09, "podge"=> 1.0166672e-07, "podgy"=> 7.719936000000001e-08, "podia"=> 4.139174e-08, "poems"=> 1.748114e-05, "poeps"=> 1.419377e-10, "poesy"=> 2.5325839999999995e-07, "poets"=> 1.0041966e-05, "pogey"=> 5.136494000000001e-09, "pogge"=> 2.201142e-07, "pogos"=> 2.39468e-09, "pohed"=> 8.171032000000001e-11, "poilu"=> 3.299284e-08, "poind"=> 1.535977e-09, "point"=> 0.000306212, "poise"=> 1.100621e-06, "pokal"=> 4.157262e-09, "poked"=> 3.868988e-06, "poker"=> 3.473986e-06, "pokes"=> 4.945919999999999e-07, "pokey"=> 1.572154e-07, "pokie"=> 1.1304258e-08, "polar"=> 8.471136000000001e-06, "poled"=> 1.123758e-07, "poler"=> 1.0120709999999999e-08, "poles"=> 8.637438e-06, "poley"=> 3.755142e-08, "polio"=> 1.274746e-06, "polis"=> 1.3991539999999999e-06, "polje"=> 4.3839399999999995e-08, "polka"=> 6.847174e-07, "polks"=> 9.354212e-09, "polls"=> 3.2849840000000003e-06, "polly"=> 4.9554620000000005e-06, "polos"=> 7.31991e-08, "polts"=> 4.964118e-10, "polyp"=> 5.593188e-07, "polys"=> 2.3284520000000002e-08, "pombe"=> 7.793736e-08, "pomes"=> 1.445522e-08, "pommy"=> 2.21971e-08, "pomos"=> 5.193896e-09, "pomps"=> 6.838019999999999e-08, "ponce"=> 5.506036e-07, "poncy"=> 2.04553e-08, "ponds"=> 2.5746119999999998e-06, "pones"=> 1.540608e-08, "poney"=> 1.4892629999999998e-08, "ponga"=> 3.5067100000000004e-08, "pongo"=> 1.4206780000000001e-07, "pongs"=> 1.2324000000000002e-08, "pongy"=> 5.587554e-09, "ponks"=> 2.1192666e-10, "ponts"=> 9.762754e-08, "ponty"=> 1.573014e-06, "ponzu"=> 2.9550620000000002e-08, "pooch"=> 1.8556799999999998e-07, "poods"=> 2.3106280000000002e-08, "pooed"=> 1.3744080000000001e-08, "poofs"=> 2.552752e-08, "poofy"=> 3.1012740000000004e-08, "poohs"=> 8.416344e-09, "pooja"=> 2.4067680000000005e-07, "pooka"=> 6.524414e-08, "pooks"=> 5.943132e-09, "pools"=> 5.773982e-06, "poons"=> 8.510251999999999e-09, "poops"=> 5.524326e-08, "poopy"=> 4.580552e-08, "poori"=> 1.2340578e-08, "poort"=> 6.924265999999999e-08, "poots"=> 2.291904e-08, "poove"=> 5.506352e-10, "poovy"=> 1.4488888e-10, "popes"=> 1.3663000000000002e-06, "poppa"=> 3.006968e-07, "poppy"=> 4.004753999999999e-06, "popsy"=> 1.730134e-08, "porae"=> 7.707622e-11, "poral"=> 1.595162e-08, "porch"=> 1.2831499999999999e-05, "pored"=> 3.90315e-07, "porer"=> 1.9806719999999998e-09, "pores"=> 3.279626e-06, "porge"=> 4.490262e-09, "porgy"=> 1.350438e-07, "porin"=> 4.797344e-08, "porks"=> 5.375984e-09, "porky"=> 1.6859120000000001e-07, "porno"=> 2.3527060000000002e-07, "porns"=> 1.941e-09, "porny"=> 7.30428e-09, "porta"=> 2.0046119999999997e-06, "ports"=> 8.807124e-06, "porty"=> 7.99645e-09, "posed"=> 1.0059392000000002e-05, "poser"=> 1.713882e-07, "poses"=> 5.687474e-06, "posey"=> 3.4199659999999997e-07, "posho"=> 6.501578e-09, "posit"=> 1.588532e-06, "posse"=> 1.178888e-06, "posts"=> 1.053856e-05, "potae"=> 6.03722e-09, "potch"=> 3.904376e-08, "poted"=> 1.654749e-10, "potes"=> 3.854514e-08, "potin"=> 2.40198e-08, "potoo"=> 4.39009e-09, "potsy"=> 1.3573336e-08, "potto"=> 1.595706e-08, "potts"=> 1.0234322000000001e-06, "potty"=> 6.18864e-07, "pouch"=> 3.4844400000000003e-06, "pouff"=> 4.40804e-09, "poufs"=> 9.9613e-09, "pouke"=> 1.9868866e-09, "pouks"=> 9.282666e-11, "poule"=> 4.361792e-08, "poulp"=> 9.065204e-09, "poult"=> 9.312334e-08, "pound"=> 1.35416e-05, "poupe"=> 4.2823316e-09, "poupt"=> 0.0, "pours"=> 1.6874980000000003e-06, "pouts"=> 1.2471180000000003e-07, "pouty"=> 2.736224e-07, "powan"=> 2.0508144e-09, "power"=> 0.0003364674, "powin"=> 4.7183644e-10, "pownd"=> 1.943248e-09, "powns"=> 9.06698e-10, "powny"=> 4.978414e-09, "powre"=> 4.6521400000000005e-08, "poxed"=> 1.1338460000000001e-08, "poxes"=> 7.709816e-09, "poynt"=> 2.5466139999999996e-08, "poyou"=> 3.2701528e-10, "poyse"=> 6.366049999999999e-10, "pozzy"=> 1.6998719999999999e-09, "praam"=> 4.1073424e-09, "prads"=> 2.9011700000000002e-09, "prahu"=> 1.8831837999999997e-08, "prams"=> 1.0618996e-07, "prana"=> 4.060934e-07, "prang"=> 6.034715999999999e-08, "prank"=> 9.589538000000002e-07, "praos"=> 2.603514e-09, "prase"=> 4.683311999999999e-09, "prate"=> 1.1027012000000002e-07, "prats"=> 7.07087e-08, "pratt"=> 2.8484639999999998e-06, "praty"=> 3.09875e-09, "praus"=> 1.6883880000000002e-08, "prawn"=> 4.876134e-07, "prays"=> 1.3100840000000002e-06, "predy"=> 1.7052174000000002e-09, "preed"=> 1.3225492e-09, "preen"=> 3.086784e-07, "prees"=> 1.658219e-08, "preif"=> 4.5618880000000003e-10, "prems"=> 5.6796939999999995e-09, "premy"=> 1.2378174000000002e-09, "prent"=> 3.034476e-08, "preon"=> 2.656908e-09, "preop"=> 5.5581740000000004e-08, "preps"=> 7.829749999999999e-08, "presa"=> 9.948892e-08, "prese"=> 1.7047402000000001e-07, "press"=> 0.0002733516, "prest"=> 2.2125399999999998e-07, "preve"=> 1.1767785999999998e-08, "prexy"=> 1.1797784e-08, "preys"=> 1.9507039999999998e-07, "prial"=> 8.655796e-09, "price"=> 9.274856e-05, "prick"=> 2.696754e-06, "pricy"=> 3.696196e-08, "pride"=> 2.837114e-05, "pried"=> 8.735562000000001e-07, "prief"=> 1.1655502e-09, "prier"=> 5.251638000000001e-08, "pries"=> 1.370188e-07, "prigs"=> 3.852928e-08, "prill"=> 2.4075e-08, "prima"=> 3.03624e-06, "prime"=> 3.1051880000000005e-05, "primi"=> 1.976782e-07, "primo"=> 9.93717e-07, "primp"=> 3.517794e-08, "prims"=> 1.9080000000000003e-08, "primy"=> 2.4429419999999996e-09, "prink"=> 9.669308000000001e-09, "print"=> 3.042962e-05, "prion"=> 5.069113999999999e-07, "prior"=> 6.218977999999998e-05, "prise"=> 3.9229159999999996e-07, "prism"=> 2.46303e-06, "priss"=> 1.0066554000000001e-07, "privy"=> 3.057196e-06, "prize"=> 1.4332400000000002e-05, "proas"=> 1.8410322e-08, "probe"=> 8.528416e-06, "probs"=> 7.767152e-08, "prods"=> 3.187806e-07, "proem"=> 1.506972e-07, "profs"=> 1.0472164000000001e-07, "progs"=> 2.183614e-08, "proin"=> 6.066652e-07, "proke"=> 1.2208634e-09, "prole"=> 7.88759e-08, "proll"=> 9.303784e-09, "promo"=> 2.581868e-07, "proms"=> 1.6579380000000002e-07, "prone"=> 9.107965999999999e-06, "prong"=> 4.867958e-07, "pronk"=> 4.837793999999999e-08, "proof"=> 3.685348e-05, "props"=> 2.2010060000000004e-06, "prore"=> 1.3548366e-09, "prose"=> 7.961707999999999e-06, "proso"=> 2.80147e-08, "pross"=> 2.1640219999999997e-07, "prost"=> 1.5641519999999997e-07, "prosy"=> 7.460866e-08, "proto"=> 2.564344e-06, "proud"=> 2.7283440000000003e-05, "proul"=> 9.698890000000001e-10, "prove"=> 4.0700299999999994e-05, "prowl"=> 5.471391999999999e-07, "prows"=> 1.013374e-07, "proxy"=> 4.990537999999999e-06, "proyn"=> 3.1934763999999997e-10, "prude"=> 2.823244e-07, "prune"=> 7.973156e-07, "prunt"=> 4.516108e-10, "pruta"=> 9.046704e-10, "pryer"=> 3.306772e-08, "pryse"=> 3.258928e-08, "psalm"=> 9.552614e-06, "pseud"=> 8.226966e-08, "pshaw"=> 2.526894e-07, "psion"=> 1.9527872e-08, "psoae"=> 4.3991286e-10, "psoai"=> 1.6027794000000001e-10, "psoas"=> 3.42748e-07, "psora"=> 1.0407699999999998e-08, "psych"=> 6.952689999999999e-07, "psyop"=> 4.1421079999999997e-08, "pubco"=> 4.144581999999999e-09, "pubes"=> 9.579044000000001e-08, "pubic"=> 1.4780499999999998e-06, "pubis"=> 4.4820860000000007e-07, "pucan"=> 2.438254e-10, "pucer"=> 3.3914e-10, "puces"=> 1.67283e-08, "pucka"=> 4.562248e-09, "pucks"=> 8.846422e-08, "puddy"=> 2.1553819999999998e-08, "pudge"=> 7.569684e-08, "pudgy"=> 4.729258e-07, "pudic"=> 2.85087e-09, "pudor"=> 3.522772e-08, "pudsy"=> 3.596288e-10, "pudus"=> 3.794035800000001e-10, "puers"=> 6.168588e-09, "puffa"=> 1.772992e-08, "puffs"=> 1.404948e-06, "puffy"=> 1.4377719999999999e-06, "puggy"=> 1.7057e-08, "pugil"=> 6.5571820000000005e-09, "puhas"=> 7.506132199999998e-10, "pujah"=> 9.400712e-10, "pujas"=> 4.2160940000000004e-08, "pukas"=> 2.043498e-09, "puked"=> 1.9052139999999998e-07, "puker"=> 5.5965739999999995e-09, "pukes"=> 3.4726220000000007e-08, "pukey"=> 1.273606e-08, "pukka"=> 5.2802640000000004e-08, "pukus"=> 7.341335999999999e-10, "pulao"=> 2.8519840000000002e-08, "pulas"=> 1.4973766e-09, "puled"=> 4.9310359999999995e-09, "puler"=> 5.450704e-10, "pules"=> 3.534204e-09, "pulik"=> 9.632034e-10, "pulis"=> 1.691788e-08, "pulka"=> 5.232158e-09, "pulks"=> 2.15332e-09, "pulli"=> 1.66489e-08, "pulls"=> 9.478994e-06, "pully"=> 1.7890759999999997e-08, "pulmo"=> 2.3027719999999996e-08, "pulps"=> 1.972554e-07, "pulpy"=> 1.77474e-07, "pulse"=> 1.973968e-05, "pulus"=> 5.97773e-10, "pumas"=> 8.898098000000001e-08, "pumie"=> 2.338874e-10, "pumps"=> 4.860682000000001e-06, "punas"=> 3.408956e-09, "punce"=> 7.519783999999999e-10, "punch"=> 9.825002e-06, "punga"=> 1.2572965999999999e-08, "pungs"=> 6.671826000000001e-09, "punji"=> 1.4371219999999999e-08, "punka"=> 6.744824e-09, "punks"=> 4.6535060000000003e-07, "punky"=> 5.965678e-08, "punny"=> 1.404028e-08, "punto"=> 3.768076e-07, "punts"=> 8.563278e-08, "punty"=> 2.651714e-09, "pupae"=> 1.9546099999999998e-07, "pupal"=> 1.1876946e-07, "pupas"=> 4.179052000000001e-09, "pupil"=> 6.701502e-06, "puppy"=> 4.66344e-06, "pupus"=> 3.4912460000000003e-09, "purda"=> 1.9719619999999997e-09, "pured"=> 2.961664e-09, "puree"=> 8.077438000000001e-07, "purer"=> 8.030641999999999e-07, "pures"=> 5.02928e-08, "purge"=> 2.169016e-06, "purin"=> 1.3150832e-08, "puris"=> 3.346504e-08, "purls"=> 1.3339296e-08, "purpy"=> 1.6093956000000002e-10, "purrs"=> 1.872828e-07, "purse"=> 1.0399974e-05, "pursy"=> 2.236496e-08, "purty"=> 1.8301480000000002e-07, "puses"=> 6.558642e-09, "pushy"=> 5.680558000000001e-07, "pusle"=> 4.40529e-10, "pussy"=> 9.24308e-06, "putid"=> 3.579802e-10, "puton"=> 9.396576e-09, "putti"=> 8.27801e-08, "putto"=> 3.6749760000000004e-08, "putts"=> 9.441864000000001e-08, "putty"=> 6.526872000000001e-07, "puzel"=> 3.980012e-09, "pwned"=> 7.421155999999999e-09, "pyats"=> 7.932657999999999e-10, "pyets"=> 2.229126e-10, "pygal"=> 3.4251444e-09, "pygmy"=> 4.3884880000000003e-07, "pyins"=> 0.0, "pylon"=> 2.6131740000000006e-07, "pyned"=> 1.0112579999999999e-09, "pynes"=> 1.3559479999999998e-08, "pyoid"=> 6.907258e-11, "pyots"=> 3.5205923999999995e-10, "pyral"=> 1.3296542000000002e-09, "pyran"=> 5.07389e-08, "pyres"=> 1.2547620000000002e-07, "pyrex"=> 1.2808680000000002e-07, "pyric"=> 3.3031700000000003e-09, "pyros"=> 2.8164644e-08, "pyxed"=> 4.307754e-11, "pyxes"=> 4.103352e-09, "pyxie"=> 1.0870086e-09, "pyxis"=> 5.247626000000001e-08, "pzazz"=> 7.587456e-10, "qadis"=> 4.4852079999999996e-08, "qaids"=> 1.8221607999999997e-09, "qajaq"=> 1.3547494000000001e-09, "qanat"=> 4.6257639999999994e-08, "qapik"=> 0.0, "qibla"=> 7.385018e-08, "qophs"=> 0.0, "qorma"=> 1.5898837999999997e-09, "quack"=> 6.514090000000001e-07, "quads"=> 2.596082e-07, "quaff"=> 9.074953999999999e-08, "quags"=> 2.584874e-09, "quail"=> 1.087754e-06, "quair"=> 2.452206e-08, "quais"=> 9.627430000000001e-08, "quake"=> 9.859632e-07, "quaky"=> 4.752058e-09, "quale"=> 5.694836000000001e-07, "qualm"=> 1.906416e-07, "quant"=> 4.841074e-07, "quare"=> 1.9344840000000002e-07, "quark"=> 8.27431e-07, "quart"=> 1.8102859999999999e-06, "quash"=> 4.1260979999999995e-07, "quasi"=> 8.224336e-06, "quass"=> 9.283312e-09, "quate"=> 4.2252439999999995e-08, "quats"=> 1.0101084000000003e-08, "quayd"=> 1.856616e-10, "quays"=> 3.478748e-07, "qubit"=> 3.782352e-07, "quean"=> 3.581374e-08, "queen"=> 4.6474040000000005e-05, "queer"=> 1.1703648e-05, "quell"=> 1.2243200000000002e-06, "queme"=> 2.8693459999999998e-09, "quena"=> 7.930514e-09, "quern"=> 5.98509e-08, "query"=> 1.0179494e-05, "quest"=> 1.2649540000000001e-05, "queue"=> 4.292309999999999e-06, "queyn"=> 1.5549606e-09, "queys"=> 3.3184720000000003e-10, "quich"=> 8.222866e-10, "quick"=> 5.46731e-05, "quids"=> 3.045634e-08, "quiet"=> 5.55162e-05, "quiff"=> 5.491922000000001e-08, "quill"=> 1.358184e-06, "quilt"=> 2.9969720000000003e-06, "quims"=> 3.0620499999999998e-09, "quina"=> 4.2222899999999995e-08, "quine"=> 9.002735999999999e-07, "quino"=> 2.7904934000000003e-08, "quins"=> 7.865031999999998e-09, "quint"=> 5.69019e-07, "quipo"=> 1.205305e-09, "quips"=> 3.198542e-07, "quipu"=> 3.552138e-08, "quire"=> 2.2549200000000004e-07, "quirk"=> 9.261552e-07, "quirt"=> 9.591085999999999e-08, "quist"=> 1.227432e-07, "quite"=> 0.0001675856, "quits"=> 7.429452e-07, "quoad"=> 4.0466740000000004e-08, "quods"=> 9.623190000000001e-10, "quoif"=> 1.2469516000000002e-09, "quoin"=> 4.4003539999999994e-08, "quoit"=> 3.973806e-08, "quoll"=> 2.31189e-08, "quonk"=> 8.369918e-10, "quops"=> 8.885242000000001e-11, "quota"=> 3.21461e-06, "quote"=> 1.0522899999999997e-05, "quoth"=> 1.1567384e-06, "qursh"=> 6.7301322e-10, "quyte"=> 7.934529999999999e-09, "rabat"=> 3.6475480000000003e-07, "rabbi"=> 5.181686e-06, "rabic"=> 5.085424e-09, "rabid"=> 8.49383e-07, "rabis"=> 4.456944e-09, "raced"=> 8.047528e-06, "racer"=> 5.638818000000001e-07, "races"=> 1.043382e-05, "rache"=> 1.2813046e-07, "racks"=> 1.726484e-06, "racon"=> 8.099080000000001e-09, "radar"=> 7.655714e-06, "radge"=> 3.2927359999999995e-09, "radii"=> 1.135962e-06, "radio"=> 3.74977e-05, "radix"=> 4.918566000000001e-07, "radon"=> 7.570229999999999e-07, "raffs"=> 4.24181e-09, "rafts"=> 8.380628e-07, "ragas"=> 7.058796e-08, "ragde"=> 1.4277716e-09, "raged"=> 2.33514e-06, "ragee"=> 9.147412e-10, "rager"=> 4.224846e-08, "rages"=> 7.073336e-07, "ragga"=> 1.454874e-08, "raggs"=> 1.0133024e-08, "raggy"=> 1.9510720000000002e-08, "ragis"=> 1.4248970000000002e-09, "ragus"=> 7.696695999999998e-09, "rahed"=> 1.6664161999999998e-10, "rahui"=> 7.395152000000001e-09, "raias"=> 7.594764e-10, "raids"=> 3.5572759999999996e-06, "raiks"=> 5.371164e-11, "raile"=> 1.4724004e-08, "rails"=> 3.3816000000000004e-06, "raine"=> 1.0355902e-06, "rains"=> 3.5919580000000006e-06, "rainy"=> 3.3627880000000003e-06, "raird"=> 5.951266000000001e-11, "raise"=> 3.4928979999999996e-05, "raita"=> 8.356154e-08, "raits"=> 3.0483079999999996e-09, "rajah"=> 6.407327999999999e-07, "rajas"=> 2.1667279999999999e-07, "rajes"=> 1.6413974e-09, "raked"=> 1.745128e-06, "rakee"=> 5.412124e-10, "raker"=> 6.508968e-08, "rakes"=> 3.860674e-07, "rakia"=> 1.5469659999999996e-08, "rakis"=> 4.8944300000000004e-09, "rakus"=> 7.459256e-09, "rales"=> 1.670824e-07, "rally"=> 4.246762e-06, "ralph"=> 1.0417968e-05, "ramal"=> 2.263274e-08, "ramee"=> 5.675744e-08, "ramen"=> 3.873902e-07, "ramet"=> 8.06949e-08, "ramie"=> 1.0504122e-07, "ramin"=> 8.316398e-08, "ramis"=> 8.835699999999999e-08, "rammy"=> 9.706664e-09, "ramps"=> 8.828956e-07, "ramus"=> 6.520315999999999e-07, "ranas"=> 2.6958379999999998e-08, "rance"=> 3.395298e-07, "ranch"=> 1.1237359999999999e-05, "rands"=> 6.232358000000001e-08, "randy"=> 3.86286e-06, "ranee"=> 5.890618e-08, "ranga"=> 1.1797149999999999e-07, "range"=> 0.00011528160000000001, "rangi"=> 1.0587655999999998e-07, "rangs"=> 1.5230985999999998e-08, "rangy"=> 1.8180019999999998e-07, "ranid"=> 8.923970000000001e-09, "ranis"=> 7.063616e-08, "ranke"=> 2.710156e-07, "ranks"=> 1.2285759999999999e-05, "rants"=> 2.891206e-07, "raped"=> 3.259352e-06, "raper"=> 1.0161728000000001e-07, "rapes"=> 7.318422e-07, "raphe"=> 2.9802660000000003e-07, "rapid"=> 3.378186e-05, "rappe"=> 3.7767520000000004e-08, "rared"=> 5.27043e-09, "raree"=> 1.56682e-08, "rarer"=> 9.659218e-07, "rares"=> 4.387106e-08, "rarks"=> 1.0210404e-10, "rased"=> 1.2241184e-08, "raser"=> 2.0989539999999997e-08, "rases"=> 3.4598140000000003e-09, "rasps"=> 1.241996e-07, "raspy"=> 8.452504e-07, "rasse"=> 5.588353999999999e-08, "rasta"=> 1.773408e-07, "ratal"=> 1.1411962000000002e-09, "ratan"=> 1.279642e-07, "ratas"=> 1.3567519999999999e-08, "ratch"=> 6.733566e-09, "rated"=> 6.600696e-06, "ratel"=> 3.268876e-08, "rater"=> 6.149576e-07, "rates"=> 6.32641e-05, "ratha"=> 1.0268982000000001e-07, "rathe"=> 4.3760019999999996e-08, "raths"=> 4.560204e-08, "ratio"=> 4.400232e-05, "ratoo"=> 2.9537700000000004e-10, "ratos"=> 7.699148e-09, "ratty"=> 4.950114e-07, "ratus"=> 1.893984e-08, "rauns"=> 1.0271304000000001e-10, "raupo"=> 4.90421e-09, "raved"=> 4.384356e-07, "ravel"=> 3.728954e-07, "raven"=> 5.645502e-06, "raver"=> 7.800240000000001e-08, "raves"=> 2.143724e-07, "ravey"=> 3.176938e-09, "ravin"=> 8.268636e-08, "rawer"=> 3.68622e-08, "rawin"=> 1.6857537999999999e-09, "rawly"=> 3.6439899999999995e-08, "rawns"=> 3.1052338e-10, "raxed"=> 2.382334e-09, "raxes"=> 1.0624986e-09, "rayah"=> 1.2422376e-08, "rayas"=> 1.944156e-08, "rayed"=> 1.983084e-07, "rayle"=> 1.78526e-08, "rayne"=> 5.650224e-07, "rayon"=> 3.1780140000000005e-07, "razed"=> 5.385002000000001e-07, "razee"=> 3.96044e-09, "razer"=> 2.9703640000000004e-08, "razes"=> 1.1402403999999998e-08, "razoo"=> 4.9011919999999995e-09, "razor"=> 3.6925120000000006e-06, "reach"=> 7.524206e-05, "react"=> 1.3260860000000001e-05, "readd"=> 1.5988046e-09, "reads"=> 1.2778599999999998e-05, "ready"=> 0.00010917720000000002, "reais"=> 5.1916979999999995e-08, "reaks"=> 2.7546719999999998e-09, "realm"=> 2.119768e-05, "realo"=> 1.3565481999999999e-08, "reals"=> 2.38086e-07, "reame"=> 4.6876380000000004e-08, "reams"=> 3.53536e-07, "reamy"=> 7.338614e-09, "reans"=> 1.5673145999999999e-09, "reaps"=> 2.0460040000000001e-07, "rearm"=> 1.0746698e-07, "rears"=> 3.1622119999999997e-07, "reast"=> 6.9529739999999994e-09, "reata"=> 1.999884e-08, "reate"=> 2.474876e-08, "reave"=> 1.4383097999999999e-08, "rebar"=> 3.3553760000000003e-07, "rebbe"=> 2.736816e-07, "rebec"=> 2.360576e-08, "rebel"=> 7.86909e-06, "rebid"=> 4.074906e-08, "rebit"=> 7.029384000000001e-10, "rebop"=> 1.8933912e-09, "rebus"=> 5.951997999999999e-07, "rebut"=> 4.779577999999999e-07, "rebuy"=> 2.8322680000000002e-08, "recal"=> 2.998334e-08, "recap"=> 7.43256e-07, "recce"=> 1.5467e-07, "recco"=> 2.58135e-08, "reccy"=> 2.354434e-09, "recit"=> 2.1573560000000004e-08, "recks"=> 2.247022e-08, "recon"=> 5.848812e-07, "recta"=> 1.5207840000000002e-07, "recti"=> 7.06092e-08, "recto"=> 2.9742539999999997e-07, "recur"=> 1.3447239999999998e-06, "recut"=> 5.903414e-08, "redan"=> 1.0266088e-07, "redds"=> 1.129889e-08, "reddy"=> 1.657158e-06, "reded"=> 5.913892e-10, "redes"=> 1.1532597999999999e-07, "redia"=> 9.728868e-09, "redid"=> 6.252437999999999e-08, "redip"=> 1.0034114e-09, "redly"=> 8.221114e-08, "redon"=> 8.704927999999998e-08, "redos"=> 5.463102e-09, "redox"=> 2.736346e-06, "redry"=> 1.463474e-09, "redub"=> 2.8135719999999996e-09, "redux"=> 3.122894e-07, "redye"=> 3.3059759999999998e-09, "reech"=> 7.631168000000001e-09, "reede"=> 7.369434e-08, "reeds"=> 1.8920179999999997e-06, "reedy"=> 4.2412619999999995e-07, "reefs"=> 2.0442640000000003e-06, "reefy"=> 1.0581558e-08, "reeks"=> 2.551426e-07, "reeky"=> 6.41e-09, "reels"=> 7.161535999999999e-07, "reens"=> 4.809196e-09, "reest"=> 1.98191e-09, "reeve"=> 1.1205344e-06, "refed"=> 5.453357999999999e-09, "refel"=> 8.869684e-10, "refer"=> 3.688776e-05, "reffo"=> 2.604328e-09, "refis"=> 7.275748e-10, "refit"=> 2.7913439999999996e-07, "refix"=> 5.763444e-09, "refly"=> 1.0914160000000001e-09, "refry"=> 2.041416e-09, "regal"=> 1.7406039999999998e-06, "regar"=> 2.513642e-08, "reges"=> 7.029450000000001e-08, "reggo"=> 5.067662e-10, "regie"=> 6.519549999999998e-08, "regma"=> 2.237312e-09, "regna"=> 7.095972000000001e-08, "regos"=> 2.6279107999999998e-08, "regur"=> 5.170958e-09, "rehab"=> 1.498362e-06, "rehem"=> 2.832704e-09, "reifs"=> 1.1144598e-09, "reify"=> 2.6437759999999997e-07, "reign"=> 1.3905259999999998e-05, "reiki"=> 6.560356e-07, "reiks"=> 3.3140716e-09, "reink"=> 7.397740000000001e-10, "reins"=> 4.435474e-06, "reird"=> 2.527001e-10, "reist"=> 6.700602e-08, "reive"=> 3.77115e-09, "rejig"=> 9.854701999999999e-09, "rejon"=> 4.579732e-09, "reked"=> 1.6188639999999999e-10, "rekes"=> 3.2990360000000003e-10, "rekey"=> 1.6544739999999997e-08, "relax"=> 1.3369420000000003e-05, "relay"=> 3.992182e-06, "relet"=> 9.660644e-09, "relic"=> 2.0789200000000002e-06, "relie"=> 1.0546211999999999e-08, "relit"=> 9.558931999999999e-08, "rello"=> 1.3721253999999999e-08, "reman"=> 9.286886e-09, "remap"=> 5.944031999999999e-08, "remen"=> 2.29522e-08, "remet"=> 9.053086e-09, "remex"=> 2.9454640000000003e-09, "remit"=> 1.549288e-06, "remix"=> 4.180062e-07, "renal"=> 1.6287019999999996e-05, "renay"=> 8.652318e-09, "rends"=> 1.1460910000000001e-07, "renew"=> 3.71405e-06, "reney"=> 1.6853939999999997e-09, "renga"=> 4.213032e-08, "renig"=> 3.1547356e-09, "renin"=> 7.621416000000001e-07, "renne"=> 6.86272e-08, "renos"=> 1.758746e-08, "rente"=> 5.5156999999999996e-08, "rents"=> 3.59179e-06, "reoil"=> 3.056465999999999e-10, "reorg"=> 2.0581296e-08, "repay"=> 3.83655e-06, "repeg"=> 2.683392e-10, "repel"=> 1.3351979999999999e-06, "repin"=> 5.657993999999999e-08, "repla"=> 2.084698e-09, "reply"=> 2.8232499999999998e-05, "repos"=> 2.289434e-07, "repot"=> 2.4696879999999998e-08, "repps"=> 1.806936e-09, "repro"=> 7.959358e-08, "reran"=> 2.4026799999999996e-08, "rerig"=> 8.4118e-10, "rerun"=> 3.3243460000000003e-07, "resat"=> 8.99422e-09, "resaw"=> 2.4193600000000002e-09, "resay"=> 6.994946e-10, "resee"=> 4.594678e-09, "reses"=> 3.562644e-09, "reset"=> 3.068364e-06, "resew"=> 2.475234e-09, "resid"=> 7.33745e-08, "resin"=> 4.011762e-06, "resit"=> 1.4255339999999998e-08, "resod"=> 9.154487999999999e-10, "resow"=> 3.0412839999999997e-09, "resto"=> 1.4392191999999998e-07, "rests"=> 7.507306e-06, "resty"=> 5.937918e-09, "resus"=> 5.49983e-08, "retag"=> 1.0543768e-09, "retax"=> 3.2059686e-10, "retch"=> 2.1267760000000003e-07, "retem"=> 2.4282976e-09, "retia"=> 1.2853775999999999e-08, "retie"=> 4.013386e-08, "retox"=> 7.548318000000001e-09, "retro"=> 1.0860886000000002e-06, "retry"=> 1.571e-07, "reuse"=> 3.7614080000000003e-06, "revel"=> 1.162544e-06, "revet"=> 1.4378938000000003e-08, "revie"=> 5.6075159999999996e-08, "revue"=> 2.4688720000000003e-06, "rewan"=> 1.7066115999999999e-09, "rewax"=> 4.4761880000000003e-10, "rewed"=> 2.4653860000000003e-09, "rewet"=> 7.2669740000000015e-09, "rewin"=> 1.468162e-09, "rewon"=> 2.1057300000000003e-09, "rewth"=> 7.44869e-10, "rexes"=> 1.4195880000000002e-08, "rezes"=> 2.723668e-10, "rheas"=> 1.84456e-08, "rheme"=> 7.079214e-08, "rheum"=> 6.829691999999999e-07, "rhies"=> 9.707704000000001e-11, "rhime"=> 9.345484e-09, "rhine"=> 2.676146e-06, "rhino"=> 8.154772000000001e-07, "rhody"=> 2.195664e-08, "rhomb"=> 1.815642e-08, "rhone"=> 3.303208e-07, "rhumb"=> 2.7054759999999996e-08, "rhyme"=> 3.20681e-06, "rhyne"=> 5.947096000000001e-08, "rhyta"=> 1.003994e-08, "riads"=> 1.1644758e-08, "rials"=> 1.0834623999999998e-07, "riant"=> 2.6974160000000002e-08, "riata"=> 4.9538099999999995e-08, "ribas"=> 1.0565014e-07, "ribby"=> 2.4052840000000002e-08, "ribes"=> 1.0610591999999999e-07, "riced"=> 2.830026e-08, "ricer"=> 2.938038e-08, "rices"=> 4.30762e-08, "ricey"=> 3.4558279999999994e-09, "richt"=> 8.060710000000001e-08, "ricin"=> 1.7287079999999998e-07, "ricks"=> 3.8659979999999994e-07, "rider"=> 6.4738939999999985e-06, "rides"=> 4.234560000000001e-06, "ridge"=> 1.2056880000000002e-05, "ridgy"=> 2.238898e-08, "ridic"=> 7.529263999999999e-09, "riels"=> 1.3375064000000001e-08, "riems"=> 3.6048240000000004e-09, "rieve"=> 9.303222000000002e-09, "rifer"=> 2.1430808e-09, "riffs"=> 2.314148e-07, "rifle"=> 1.25484e-05, "rifte"=> 3.5286904e-10, "rifts"=> 4.410132e-07, "rifty"=> 6.5126486e-10, "riggs"=> 8.532316e-07, "right"=> 0.0005686142, "rigid"=> 1.30983e-05, "rigol"=> 8.947477999999999e-09, "rigor"=> 1.975956e-06, "riled"=> 5.138038e-07, "riles"=> 1.198076e-07, "riley"=> 7.786080000000002e-06, "rille"=> 1.3522762e-08, "rills"=> 1.46872e-07, "rimae"=> 2.00757e-09, "rimed"=> 4.6302939999999995e-08, "rimer"=> 7.808023999999999e-08, "rimes"=> 1.9144480000000002e-07, "rimus"=> 1.4248402e-09, "rinds"=> 1.778698e-07, "rindy"=> 1.0707155999999998e-08, "rines"=> 2.181326e-08, "rings"=> 1.367292e-05, "rinks"=> 1.0483238e-07, "rinse"=> 2.34205e-06, "rioja"=> 1.995178e-07, "riots"=> 4.308412e-06, "riped"=> 3.3070780000000003e-09, "ripen"=> 6.120288e-07, "riper"=> 1.874004e-07, "ripes"=> 3.196078e-09, "ripps"=> 1.0337258e-08, "risen"=> 9.110026e-06, "riser"=> 6.537878e-07, "rises"=> 1.048042e-05, "rishi"=> 3.822038e-07, "risks"=> 3.53737e-05, "risky"=> 6.846318000000001e-06, "risps"=> 6.45248e-10, "risus"=> 5.481357999999999e-07, "rites"=> 5.4744879999999995e-06, "ritts"=> 1.559754e-08, "ritzy"=> 1.196464e-07, "rival"=> 9.911916000000001e-06, "rivas"=> 3.7504220000000004e-07, "rived"=> 3.6354099999999995e-08, "rivel"=> 5.50494e-09, "riven"=> 4.856348e-07, "river"=> 9.457312e-05, "rives"=> 1.859332e-07, "rivet"=> 5.092374000000001e-07, "riyal"=> 3.0236379999999996e-08, "rizas"=> 2.010762e-09, "roach"=> 1.19625e-06, "roads"=> 1.922456e-05, "roams"=> 2.2150340000000002e-07, "roans"=> 1.648684e-08, "roars"=> 1.0140926e-06, "roary"=> 9.823645999999999e-09, "roast"=> 5.0190420000000005e-06, "roate"=> 8.478976000000002e-10, "robed"=> 1.0726897999999998e-06, "robes"=> 4.551796e-06, "robin"=> 1.075388e-05, "roble"=> 4.98455e-08, "robot"=> 1.0292994e-05, "rocks"=> 2.2627959999999998e-05, "rocky"=> 1.0073332000000001e-05, "roded"=> 6.664838e-09, "rodeo"=> 1.498148e-06, "rodes"=> 8.758736e-08, "roger"=> 1.580668e-05, "rogue"=> 3.695584e-06, "roguy"=> 4.362962e-10, "rohes"=> 1.2455058e-09, "roids"=> 1.49636e-08, "roils"=> 3.794058e-08, "roily"=> 6.839268e-09, "roins"=> 2.9184306e-10, "roist"=> 7.794414e-10, "rojak"=> 8.739828e-09, "rojis"=> 5.2851486e-10, "roked"=> 3.3578298e-10, "roker"=> 5.1562599999999995e-08, "rokes"=> 1.5794480000000004e-09, "rolag"=> 1.2421912e-09, "roles"=> 3.526682e-05, "rolfs"=> 1.56924e-08, "rolls"=> 9.476252e-06, "romal"=> 3.630068e-09, "roman"=> 4.7381219999999994e-05, "romeo"=> 2.83379e-06, "romps"=> 7.663628000000001e-08, "ronde"=> 1.673494e-07, "rondo"=> 2.2986599999999998e-07, "roneo"=> 4.417806e-09, "rones"=> 1.1638204e-08, "ronin"=> 2.8213180000000007e-07, "ronne"=> 2.72071e-08, "ronte"=> 1.867424e-09, "ronts"=> 9.046574e-10, "roods"=> 2.793992e-08, "roofs"=> 3.8817640000000005e-06, "roofy"=> 1.3797018e-09, "rooks"=> 4.2108539999999996e-07, "rooky"=> 1.367767e-08, "rooms"=> 3.467052e-05, "roomy"=> 4.4344219999999994e-07, "roons"=> 2.2090566e-09, "roops"=> 4.251956000000001e-09, "roopy"=> 1.620524e-09, "roosa"=> 4.696391999999999e-08, "roose"=> 9.490168000000002e-08, "roost"=> 8.184016000000001e-07, "roots"=> 2.601086e-05, "rooty"=> 2.7043799999999996e-08, "roped"=> 5.971838e-07, "roper"=> 1.201114e-06, "ropes"=> 5.20502e-06, "ropey"=> 5.502624e-08, "roque"=> 3.696796e-07, "roral"=> 6.219046e-10, "rores"=> 1.8792199999999996e-09, "roric"=> 1.1472128000000001e-08, "rorid"=> 1.22019342e-09, "rorie"=> 1.0142431999999999e-07, "rorts"=> 3.016416e-09, "rorty"=> 1.0202602e-06, "rosed"=> 4.703306e-09, "roses"=> 7.633272e-06, "roset"=> 1.235343e-08, "roshi"=> 1.2312660000000002e-07, "rosin"=> 2.9998000000000007e-07, "rosit"=> 4.4530499999999994e-10, "rosti"=> 2.406952e-08, "rosts"=> 9.19655e-10, "rotal"=> 2.5179784000000003e-09, "rotan"=> 2.292847e-08, "rotas"=> 5.230568e-08, "rotch"=> 3.983472e-08, "roted"=> 2.272888e-09, "rotes"=> 2.051766e-08, "rotis"=> 5.791984e-08, "rotls"=> 1.8499064e-09, "roton"=> 1.3926862e-08, "rotor"=> 3.8040520000000005e-06, "rotos"=> 1.1435950000000001e-08, "rotte"=> 3.269398000000001e-08, "rouen"=> 1.0530266e-06, "roues"=> 1.7201922e-08, "rouge"=> 3.0972699999999996e-06, "rough"=> 2.571588e-05, "roule"=> 5.4514999999999995e-08, "rouls"=> 1.4183984e-09, "roums"=> 2.2238998e-10, "round"=> 9.37726e-05, "roups"=> 1.666296e-08, "roupy"=> 1.3706046000000002e-09, "rouse"=> 2.27501e-06, "roust"=> 6.72752e-08, "route"=> 3.2866119999999995e-05, "routh"=> 1.549478e-07, "routs"=> 6.949362e-08, "roved"=> 3.134604e-07, "roven"=> 1.0918642e-08, "rover"=> 2.4343619999999997e-06, "roves"=> 3.711518e-08, "rowan"=> 3.0706819999999996e-06, "rowdy"=> 1.248434e-06, "rowed"=> 1.184496e-06, "rowel"=> 2.941074e-08, "rowen"=> 2.6761979999999997e-07, "rower"=> 1.646084e-07, "rowie"=> 1.4428825999999999e-08, "rowme"=> 3.951479999999999e-09, "rownd"=> 6.837422000000001e-09, "rowth"=> 2.7807159999999997e-08, "rowts"=> 2.199018e-10, "royal"=> 5.1411060000000004e-05, "royne"=> 2.3413300000000002e-08, "royst"=> 2.8551678000000004e-10, "rozet"=> 1.479119e-08, "rozit"=> 1.7215312e-10, "ruana"=> 6.042335999999999e-09, "rubai"=> 3.4447459999999997e-09, "rubby"=> 1.1745744e-08, "rubel"=> 1.0248744e-07, "rubes"=> 6.415462e-08, "rubin"=> 2.586104e-06, "ruble"=> 3.584742e-07, "rubli"=> 2.8131088000000005e-09, "rubus"=> 1.341126e-07, "ruche"=> 2.2587899999999998e-08, "rucks"=> 3.081374e-08, "rudas"=> 2.4487699999999998e-08, "rudds"=> 1.8985412e-09, "ruddy"=> 1.6385120000000002e-06, "ruder"=> 1.71637e-07, "rudes"=> 1.757392e-08, "rudie"=> 2.4243819999999997e-08, "rudis"=> 3.5198039999999995e-08, "rueda"=> 1.9820940000000002e-07, "ruers"=> 2.4588260000000003e-09, "ruffe"=> 1.3395539999999999e-08, "ruffs"=> 9.703592000000001e-08, "rugae"=> 4.993348000000001e-08, "rugal"=> 1.1021314e-08, "rugby"=> 2.8315120000000002e-06, "ruggy"=> 2.1406634e-09, "ruing"=> 2.7224239999999997e-08, "ruins"=> 8.418784000000001e-06, "rukhs"=> 1.9918344e-09, "ruled"=> 1.418952e-05, "ruler"=> 1.141374e-05, "rules"=> 9.624733999999999e-05, "rumal"=> 5.398084e-09, "rumba"=> 1.812032e-07, "rumbo"=> 2.7099519999999997e-08, "rumen"=> 4.4690199999999996e-07, "rumes"=> 3.9376660000000005e-09, "rumly"=> 4.302002e-10, "rummy"=> 1.7445e-07, "rumor"=> 3.167128e-06, "rumpo"=> 6.500216e-09, "rumps"=> 6.999578e-08, "rumpy"=> 7.905894e-09, "runch"=> 4.5364339999999994e-09, "runds"=> 1.1534055999999999e-09, "runed"=> 6.152070000000001e-09, "runes"=> 9.310737999999999e-07, "rungs"=> 6.722504e-07, "runic"=> 2.557192e-07, "runny"=> 5.410030000000001e-07, "runts"=> 4.5716560000000004e-08, "runty"=> 2.9256879999999997e-08, "rupee"=> 5.229967999999999e-07, "rupia"=> 3.62097e-09, "rural"=> 3.9721520000000004e-05, "rurps"=> 3.8496302e-10, "rurus"=> 2.717525e-10, "rusas"=> 2.754022e-09, "ruses"=> 1.420516e-07, "rushy"=> 2.998556e-08, "rusks"=> 5.158305999999999e-08, "rusma"=> 4.1098499999999995e-10, "russe"=> 2.367384e-07, "rusts"=> 1.0037642e-07, "rusty"=> 4.389946e-06, "ruths"=> 1.95851e-08, "rutin"=> 1.1829988000000001e-07, "rutty"=> 3.78258e-08, "ryals"=> 2.5301920000000005e-08, "rybat"=> 2.19883224e-09, "ryked"=> 1.4271148000000002e-10, "rykes"=> 1.7881764000000002e-09, "rymme"=> 1.0462596e-10, "rynds"=> 2.440974e-10, "ryots"=> 4.2825399999999996e-08, "ryper"=> 2.4402386e-09, "saags"=> 1.2157398e-10, "sabal"=> 3.9883019999999994e-08, "sabed"=> 1.0902588e-08, "saber"=> 1.2496805999999998e-06, "sabes"=> 1.0585186000000001e-07, "sabha"=> 8.745092e-07, "sabin"=> 4.853546e-07, "sabir"=> 7.50708e-08, "sable"=> 1.0737486e-06, "sabot"=> 8.822746e-08, "sabra"=> 2.6052360000000004e-07, "sabre"=> 7.929218e-07, "sacks"=> 2.7925999999999997e-06, "sacra"=> 2.1443984e-06, "saddo"=> 6.341658e-09, "sades"=> 8.97021e-09, "sadhe"=> 4.680595999999999e-09, "sadhu"=> 1.638622e-07, "sadis"=> 6.001523999999999e-09, "sadly"=> 9.628452000000001e-06, "sados"=> 4.624121999999999e-10, "sadza"=> 1.389916e-08, "safed"=> 1.1322543999999999e-07, "safer"=> 7.3053959999999994e-06, "safes"=> 2.490198e-07, "sagas"=> 5.19163e-07, "sager"=> 1.995458e-07, "sages"=> 1.9297880000000002e-06, "saggy"=> 1.44038e-07, "sagos"=> 6.738272e-10, "sagum"=> 8.062463999999999e-09, "saheb"=> 1.5424400000000001e-07, "sahib"=> 1.570098e-06, "saice"=> 2.464088e-09, "saick"=> 6.123353999999999e-11, "saics"=> 8.902110000000001e-10, "saids"=> 1.2804738e-08, "saiga"=> 5.470272e-08, "sails"=> 3.972954e-06, "saims"=> 4.3178239999999997e-10, "saine"=> 2.9299559999999996e-08, "sains"=> 8.196060000000001e-08, "saint"=> 2.562898e-05, "sairs"=> 9.956358e-10, "saist"=> 3.840838000000001e-09, "saith"=> 7.361792000000001e-06, "sajou"=> 5.924068e-10, "sakai"=> 5.343245999999999e-07, "saker"=> 9.34895e-08, "sakes"=> 9.59115e-07, "sakia"=> 1.0637018e-08, "sakis"=> 2.8567779999999997e-08, "sakti"=> 1.7056514e-07, "salad"=> 1.0547172000000001e-05, "salal"=> 3.0114899999999997e-08, "salat"=> 1.7646360000000002e-07, "salep"=> 1.3907808e-08, "sales"=> 4.4089700000000006e-05, "salet"=> 1.8650500000000002e-08, "salic"=> 8.036676e-08, "salix"=> 1.88141e-07, "salle"=> 8.505726e-07, "sally"=> 1.062087e-05, "salmi"=> 1.044113e-07, "salol"=> 5.420828e-09, "salon"=> 3.841568e-06, "salop"=> 6.415369999999999e-08, "salpa"=> 7.579261999999999e-09, "salps"=> 1.833702e-08, "salsa"=> 1.7680580000000004e-06, "salse"=> 5.5564e-09, "salto"=> 1.0398684000000002e-07, "salts"=> 4.27441e-06, "salty"=> 2.842772e-06, "salue"=> 2.299736e-08, "salut"=> 1.841246e-07, "salve"=> 9.017162e-07, "salvo"=> 6.204664e-07, "saman"=> 1.2486246e-07, "samas"=> 2.499588e-08, "samba"=> 5.614192e-07, "sambo"=> 2.3658419999999998e-07, "samek"=> 3.578152e-08, "samel"=> 1.6437006e-08, "samen"=> 1.1874637999999999e-07, "sames"=> 1.9124220000000004e-08, "samey"=> 1.0790428e-08, "samfu"=> 6.830736e-10, "sammy"=> 3.0944960000000005e-06, "sampi"=> 4.996594e-09, "samps"=> 3.122388e-09, "sands"=> 4.506536e-06, "sandy"=> 1.1076119999999999e-05, "saned"=> 7.082476000000001e-10, "saner"=> 1.8744880000000002e-07, "sanes"=> 3.47133e-08, "sanga"=> 1.3423286e-07, "sangh"=> 2.5679459999999997e-07, "sango"=> 9.155787999999999e-08, "sangs"=> 3.1068059999999997e-08, "sanko"=> 1.4261220000000002e-08, "sansa"=> 7.039368e-08, "santo"=> 2.165122e-06, "sants"=> 5.34251e-08, "saola"=> 1.5961687999999997e-08, "sapan"=> 1.5596290000000003e-08, "sapid"=> 1.1711826e-08, "sapor"=> 5.0742660000000004e-08, "sappy"=> 2.596862e-07, "saran"=> 2.250436e-07, "sards"=> 9.160002000000001e-09, "sared"=> 1.1905098e-09, "saree"=> 1.61485e-07, "sarge"=> 5.950063999999999e-07, "sargo"=> 4.85253e-09, "sarin"=> 3.0864280000000004e-07, "saris"=> 2.726794e-07, "sarks"=> 1.4427312e-08, "sarky"=> 1.266256e-08, "sarod"=> 1.68596e-08, "saros"=> 4.560498e-08, "sarus"=> 1.996586e-08, "saser"=> 2.1529018e-09, "sasin"=> 5.901906e-09, "sasse"=> 1.0352015999999999e-07, "sassy"=> 7.229370000000001e-07, "satai"=> 4.038561999999999e-09, "satay"=> 1.0471603999999998e-07, "sated"=> 7.245434e-07, "satem"=> 5.0603120000000005e-09, "sates"=> 8.222678000000001e-08, "satin"=> 3.2106699999999996e-06, "satis"=> 2.0983239999999996e-07, "satyr"=> 4.89088e-07, "sauba"=> 7.27095e-10, "sauce"=> 1.4959740000000001e-05, "sauch"=> 3.853474e-09, "saucy"=> 6.932687999999999e-07, "saugh"=> 1.9170378e-08, "sauls"=> 8.878579999999999e-08, "sault"=> 2.119068e-07, "sauna"=> 7.063816000000001e-07, "saunt"=> 2.4100059999999997e-08, "saury"=> 1.5164900000000002e-08, "saute"=> 1.0591468000000001e-07, "sauts"=> 3.958736e-09, "saved"=> 3.0085079999999996e-05, "saver"=> 5.558797999999999e-07, "saves"=> 3.427508e-06, "savey"=> 3.94542e-09, "savin"=> 2.93763e-07, "savor"=> 1.28431e-06, "savoy"=> 1.257392e-06, "savvy"=> 2.09843e-06, "sawah"=> 2.2292240000000002e-08, "sawed"=> 6.464298000000001e-07, "sawer"=> 4.9426519999999996e-08, "saxes"=> 1.688782e-08, "sayed"=> 3.384398e-07, "sayer"=> 4.79447e-07, "sayid"=> 9.115356e-08, "sayne"=> 6.791982e-09, "sayon"=> 8.522018000000001e-09, "sayst"=> 3.421654e-08, "sazes"=> 3.822815819999999e-09, "scabs"=> 3.805574e-07, "scads"=> 5.783772e-08, "scaff"=> 2.312126e-08, "scags"=> 1.3391588e-09, "scail"=> 4.779503e-10, "scala"=> 1.14523e-06, "scald"=> 2.44763e-07, "scale"=> 8.252088000000001e-05, "scall"=> 4.5736040000000006e-08, "scalp"=> 4.130116e-06, "scaly"=> 8.220850000000002e-07, "scamp"=> 3.188158e-07, "scams"=> 5.190771999999999e-07, "scand"=> 9.661894e-07, "scans"=> 3.5821300000000004e-06, "scant"=> 2.6677979999999996e-06, "scapa"=> 1.6151299999999997e-07, "scape"=> 3.7609520000000004e-07, "scapi"=> 3.347267e-09, "scare"=> 5.667022e-06, "scarf"=> 4.2326840000000006e-06, "scarp"=> 1.8692099999999998e-07, "scars"=> 5.301062e-06, "scart"=> 1.6789580000000002e-08, "scary"=> 5.724252e-06, "scath"=> 1.427036e-08, "scats"=> 5.5500939999999996e-08, "scatt"=> 7.81155e-09, "scaud"=> 8.169686000000001e-10, "scaup"=> 2.8393219999999996e-08, "scaur"=> 1.939364e-08, "scaws"=> 2.1612302e-10, "sceat"=> 2.13315e-09, "scena"=> 8.044098e-08, "scend"=> 9.161662000000001e-09, "scene"=> 6.300606e-05, "scent"=> 1.45058e-05, "schav"=> 1.0898948000000001e-09, "schmo"=> 1.2687254e-08, "schul"=> 4.227944e-08, "schwa"=> 1.326704e-07, "scion"=> 5.979290000000001e-07, "sclim"=> 4.151054e-10, "scody"=> 1.0254994e-10, "scoff"=> 5.857041999999999e-07, "scogs"=> 2.2744539999999998e-09, "scold"=> 1.1376898e-06, "scone"=> 3.910542e-07, "scoog"=> 1.6245866e-10, "scoop"=> 2.8172059999999996e-06, "scoot"=> 5.508192e-07, "scopa"=> 2.0011459999999997e-08, "scope"=> 3.4415240000000005e-05, "scops"=> 3.401652e-08, "score"=> 3.140676e-05, "scorn"=> 3.7863999999999994e-06, "scots"=> 3.6899e-06, "scoug"=> 2.6588063999999997e-10, "scoup"=> 5.479722e-10, "scour"=> 8.942193999999999e-07, "scout"=> 4.730444e-06, "scowl"=> 2.4087179999999996e-06, "scowp"=> 1.6015564e-10, "scows"=> 4.7827119999999994e-08, "scrab"=> 1.2754360000000003e-09, "scrae"=> 2.899218e-10, "scrag"=> 3.837982e-08, "scram"=> 3.19528e-07, "scran"=> 1.2121359999999999e-08, "scrap"=> 4.415172e-06, "scrat"=> 9.398989999999999e-09, "scraw"=> 2.9828019999999995e-09, "scray"=> 1.4501004e-09, "scree"=> 3.4323880000000004e-07, "screw"=> 7.77578e-06, "scrim"=> 1.0036784e-07, "scrip"=> 3.8725420000000007e-07, "scrob"=> 1.9156482e-09, "scrod"=> 8.448722e-09, "scrog"=> 2.51627e-09, "scrow"=> 1.5907882e-09, "scrub"=> 3.165046e-06, "scrum"=> 1.1539364e-06, "scuba"=> 7.682154e-07, "scudi"=> 1.1969676e-07, "scudo"=> 3.297282e-08, "scuds"=> 4.4513360000000005e-08, "scuff"=> 1.936734e-07, "scuft"=> 2.0998222000000002e-10, "scugs"=> 1.8097071999999999e-09, "sculk"=> 1.0338462000000001e-09, "scull"=> 2.4286080000000004e-07, "sculp"=> 2.1587640000000002e-08, "sculs"=> 5.150866000000001e-10, "scums"=> 1.1315494000000001e-08, "scups"=> 1.0165512e-09, "scurf"=> 3.489952e-08, "scurs"=> 2.590016e-09, "scuse"=> 9.964524e-08, "scuta"=> 3.094375e-08, "scute"=> 2.6716459999999997e-08, "scuts"=> 6.143898000000001e-09, "scuzz"=> 1.0761786e-08, "scyes"=> 9.909962e-11, "sdayn"=> 0.0, "sdein"=> 4.591056e-11, "seals"=> 4.798221999999999e-06, "seame"=> 5.867563999999999e-09, "seams"=> 1.852574e-06, "seamy"=> 8.778046e-08, "seans"=> 6.892702000000001e-09, "seare"=> 7.3684580000000014e-09, "sears"=> 1.4816300000000001e-06, "sease"=> 1.1454221999999999e-08, "seats"=> 1.683974e-05, "seaze"=> 1.1969642e-09, "sebum"=> 1.724322e-07, "secco"=> 6.061676e-08, "sechs"=> 9.805116e-08, "sects"=> 2.4467640000000003e-06, "sedan"=> 1.91692e-06, "seder"=> 2.84143e-07, "sedes"=> 5.639028e-08, "sedge"=> 3.551718e-07, "sedgy"=> 2.210932e-08, "sedum"=> 8.192086e-08, "seeds"=> 1.9116479999999998e-05, "seedy"=> 6.52635e-07, "seeks"=> 1.454332e-05, "seeld"=> 2.439898e-10, "seels"=> 7.355238e-09, "seely"=> 1.721982e-07, "seems"=> 0.00012460299999999999, "seeps"=> 4.81445e-07, "seepy"=> 1.0731408e-08, "seers"=> 6.591738e-07, "sefer"=> 4.121222e-07, "segar"=> 5.7445539999999996e-08, "segni"=> 9.52816e-08, "segno"=> 8.756354e-08, "segol"=> 8.009888e-09, "segos"=> 2.6415477999999995e-10, "segue"=> 2.994798e-07, "sehri"=> 3.040238e-09, "seifs"=> 1.653952e-09, "seils"=> 6.981322e-09, "seine"=> 3.4014880000000005e-06, "seirs"=> 6.641358e-09, "seise"=> 1.0648482e-08, "seism"=> 1.811232e-08, "seity"=> 1.6051428e-09, "seiza"=> 1.405094e-08, "seize"=> 6.1386e-06, "sekos"=> 3.2297328e-09, "sekts"=> 6.638768e-10, "selah"=> 7.4446e-07, "seles"=> 2.153614e-08, "selfs"=> 1.0776366000000001e-07, "sella"=> 2.855718e-07, "selle"=> 9.970001999999998e-08, "sells"=> 4.8278e-06, "selva"=> 2.15625e-07, "semee"=> 9.409848e-10, "semen"=> 2.141242e-06, "semes"=> 4.7694880000000006e-08, "semie"=> 1.1789763999999999e-09, "semis"=> 1.0367132e-07, "senas"=> 1.0138013999999999e-08, "sends"=> 8.696808e-06, "senes"=> 1.8992919999999998e-08, "sengi"=> 1.0208014e-08, "senna"=> 3.176532e-07, "senor"=> 6.463964000000001e-07, "sensa"=> 4.350096e-08, "sense"=> 0.0002067476, "sensi"=> 7.633292e-08, "sente"=> 6.481846e-08, "senti"=> 6.678698e-08, "sents"=> 5.933720000000001e-08, "senvy"=> 5.366202e-11, "senza"=> 3.573116e-07, "sepad"=> 3.84801e-09, "sepal"=> 8.904e-08, "sepia"=> 4.176734e-07, "sepic"=> 1.8469354e-08, "sepoy"=> 1.851096e-07, "septa"=> 4.374896e-07, "septs"=> 2.4424940000000003e-08, "serac"=> 2.141196e-08, "serai"=> 1.3845559999999998e-07, "seral"=> 2.6286480000000002e-08, "sered"=> 4.0268180000000005e-08, "serer"=> 2.492392e-08, "seres"=> 1.1445518e-07, "serfs"=> 8.963898000000001e-07, "serge"=> 1.439768e-06, "seric"=> 5.526758e-09, "serif"=> 2.97927e-07, "serin"=> 5.338946e-08, "serks"=> 2.1942536000000003e-09, "seron"=> 4.36897e-08, "serow"=> 1.652226e-08, "serra"=> 8.973235999999999e-07, "serre"=> 1.828226e-07, "serrs"=> 9.481448000000001e-09, "serry"=> 1.1591774e-08, "serum"=> 1.2141159999999999e-05, "serve"=> 6.585054e-05, "servo"=> 8.459796e-07, "sesey"=> 1.3670524e-10, "sessa"=> 8.180760000000002e-08, "setae"=> 3.35854e-07, "setal"=> 2.5252479999999998e-08, "seton"=> 6.02151e-07, "setts"=> 1.0471346000000001e-07, "setup"=> 7.84372e-06, "seven"=> 8.068808e-05, "sever"=> 1.224274e-06, "sewan"=> 3.2173959999999996e-09, "sewar"=> 8.104608000000002e-10, "sewed"=> 8.454666000000001e-07, "sewel"=> 2.416538e-08, "sewen"=> 2.8096120000000004e-09, "sewer"=> 1.90523e-06, "sewin"=> 2.797562e-08, "sexed"=> 3.631336e-07, "sexer"=> 3.8629579999999994e-09, "sexes"=> 3.882188e-06, "sexto"=> 4.7499939999999996e-08, "sexts"=> 1.787008e-08, "seyen"=> 4.2360299999999995e-09, "shack"=> 2.478668e-06, "shade"=> 1.405918e-05, "shads"=> 1.8274034e-08, "shady"=> 2.49275e-06, "shaft"=> 9.954926000000001e-06, "shags"=> 4.301768e-08, "shahs"=> 5.9595080000000004e-08, "shake"=> 2.1732e-05, "shako"=> 6.10853e-08, "shakt"=> 4.4475799999999995e-10, "shaky"=> 3.867618e-06, "shale"=> 2.7609860000000004e-06, "shall"=> 0.00025302040000000004, "shalm"=> 2.1678964e-09, "shalt"=> 9.08972e-06, "shaly"=> 3.551632e-08, "shama"=> 1.0564092000000001e-07, "shame"=> 2.4099940000000002e-05, "shams"=> 4.515978e-07, "shand"=> 1.9713800000000003e-07, "shank"=> 7.872934000000001e-07, "shans"=> 2.897752e-08, "shape"=> 6.425162000000001e-05, "shaps"=> 6.809544e-09, "shard"=> 8.570835999999999e-07, "share"=> 9.595331999999999e-05, "shark"=> 3.779554e-06, "sharn"=> 2.0259209999999997e-08, "sharp"=> 4.04316e-05, "shash"=> 1.525158e-08, "shaul"=> 2.458352e-07, "shave"=> 2.27731e-06, "shawl"=> 2.60439e-06, "shawm"=> 3.1270820000000005e-08, "shawn"=> 2.518088e-06, "shaws"=> 8.157096e-08, "shaya"=> 4.242994e-08, "shays"=> 1.510246e-07, "shchi"=> 5.792072e-09, "sheaf"=> 9.620708e-07, "sheal"=> 1.1072342e-08, "shear"=> 1.0598022e-05, "sheas"=> 3.995554e-09, "sheds"=> 2.519544e-06, "sheel"=> 2.0287540000000002e-08, "sheen"=> 1.911644e-06, "sheep"=> 1.891072e-05, "sheer"=> 1.081744e-05, "sheet"=> 3.0039440000000002e-05, "sheik"=> 6.041332e-07, "shelf"=> 1.1193639999999998e-05, "shell"=> 2.040042e-05, "shend"=> 7.039146e-09, "shent"=> 1.954446e-08, "sheol"=> 5.403368e-07, "sherd"=> 1.877696e-07, "shere"=> 2.1712620000000002e-07, "shero"=> 2.6498439999999995e-08, "shets"=> 4.354174e-09, "sheva"=> 1.347558e-07, "shewn"=> 4.827064e-07, "shews"=> 2.7447839999999997e-07, "shiai"=> 3.8073680000000005e-09, "shied"=> 6.744752000000001e-07, "shiel"=> 8.372018e-08, "shier"=> 5.071118e-08, "shies"=> 1.0155636e-07, "shift"=> 4.788596e-05, "shill"=> 7.292144e-08, "shily"=> 8.557904e-10, "shims"=> 8.526107999999998e-08, "shine"=> 7.714616000000001e-06, "shins"=> 6.305124e-07, "shiny"=> 6.531302e-06, "ships"=> 2.82643e-05, "shire"=> 8.517304000000002e-07, "shirk"=> 5.756838e-07, "shirr"=> 8.492523999999999e-09, "shirs"=> 4.0228380000000005e-10, "shirt"=> 3.785631999999999e-05, "shish"=> 8.546134e-08, "shiso"=> 5.214852e-08, "shist"=> 3.1687180000000004e-09, "shite"=> 3.4708020000000005e-07, "shits"=> 2.9514380000000005e-07, "shiur"=> 5.850694000000001e-09, "shiva"=> 1.78066e-06, "shive"=> 2.271832e-08, "shivs"=> 1.0688506e-08, "shlep"=> 2.76286e-09, "shlub"=> 1.1730326e-09, "shmek"=> 5.977906e-10, "shmoe"=> 1.936662e-09, "shoal"=> 6.679824e-07, "shoat"=> 2.122054e-08, "shock"=> 3.53832e-05, "shoed"=> 3.601698e-08, "shoer"=> 9.569538e-09, "shoes"=> 2.6357619999999998e-05, "shogi"=> 2.1606750000000003e-08, "shogs"=> 4.1066919999999994e-10, "shoji"=> 1.599276e-07, "shojo"=> 3.118172e-08, "shola"=> 4.505868e-08, "shone"=> 8.174736000000002e-06, "shook"=> 7.269082e-05, "shool"=> 1.173893e-08, "shoon"=> 5.142468e-08, "shoos"=> 4.941219999999999e-08, "shoot"=> 1.9441540000000003e-05, "shope"=> 5.597962e-08, "shops"=> 1.292578e-05, "shore"=> 2.205852e-05, "shorl"=> 9.664782e-10, "shorn"=> 7.964856e-07, "short"=> 0.0001699272, "shote"=> 1.1692906e-08, "shots"=> 1.1531940000000002e-05, "shott"=> 5.4664680000000004e-08, "shout"=> 9.894846e-06, "shove"=> 3.62589e-06, "showd"=> 3.574562e-09, "shown"=> 0.00013309480000000003, "shows"=> 0.0001091346, "showy"=> 7.52857e-07, "shoyu"=> 1.0589830000000002e-07, "shred"=> 1.3614919999999998e-06, "shrew"=> 6.747249999999999e-07, "shris"=> 3.9487179999999996e-10, "shrow"=> 1.1719558e-09, "shrub"=> 1.552484e-06, "shrug"=> 5.191166000000001e-06, "shtik"=> 1.961108e-09, "shtum"=> 9.225972e-09, "shtup"=> 1.930494e-09, "shuck"=> 2.173064e-07, "shule"=> 2.041502e-08, "shuln"=> 1.8908388e-09, "shuls"=> 7.927486e-09, "shuns"=> 1.8722820000000002e-07, "shunt"=> 2.0004e-06, "shura"=> 3.6649700000000003e-07, "shush"=> 4.920432e-07, "shute"=> 2.224382e-07, "shuts"=> 1.420434e-06, "shwas"=> 3.61273e-10, "shyer"=> 5.7753539999999995e-08, "shyly"=> 1.5543999999999999e-06, "sials"=> 6.502429999999999e-10, "sibbs"=> 1.6046920600000003e-09, "sibyl"=> 6.958064e-07, "sices"=> 6.678274000000001e-09, "sicht"=> 1.1321277999999999e-07, "sicko"=> 8.07748e-08, "sicks"=> 9.731998e-09, "sicky"=> 8.087753999999999e-09, "sidas"=> 1.1461804e-08, "sided"=> 6.911014e-06, "sider"=> 2.8177840000000005e-07, "sides"=> 4.57383e-05, "sidha"=> 1.1900894000000001e-08, "sidhe"=> 2.678436e-07, "sidle"=> 1.6121140000000002e-07, "siege"=> 6.839466e-06, "sield"=> 5.391682e-10, "siens"=> 1.880242e-08, "sient"=> 8.068218e-09, "sieth"=> 9.842365999999999e-10, "sieur"=> 3.231252e-07, "sieve"=> 1.7965159999999999e-06, "sifts"=> 8.321851999999999e-08, "sighs"=> 3.621454e-06, "sight"=> 6.38172e-05, "sigil"=> 2.7762279999999996e-07, "sigla"=> 3.8152960000000005e-08, "sigma"=> 2.6622099999999995e-06, "signa"=> 1.576592e-07, "signs"=> 4.855695999999999e-05, "sijos"=> 1.8785024e-10, "sikas"=> 9.685704e-10, "siker"=> 1.9866160000000002e-08, "sikes"=> 4.849741999999999e-07, "silds"=> 6.144896e-11, "siled"=> 8.872135999999999e-10, "silen"=> 1.809816e-08, "siler"=> 5.7326660000000005e-08, "siles"=> 4.6414459999999996e-08, "silex"=> 7.9703e-08, "silks"=> 1.0273501999999999e-06, "silky"=> 2.7380640000000004e-06, "sills"=> 4.68482e-07, "silly"=> 1.2805459999999999e-05, "silos"=> 8.881694e-07, "silts"=> 1.2481919999999999e-07, "silty"=> 3.3815039999999996e-07, "silva"=> 4.365448e-06, "simar"=> 3.741018e-08, "simas"=> 3.575216e-08, "simba"=> 2.0181339999999998e-07, "simis"=> 9.311215999999999e-09, "simps"=> 6.777376e-09, "simul"=> 3.3313040000000004e-07, "since"=> 0.0003521618, "sinds"=> 1.097142e-08, "sined"=> 2.2704580000000002e-09, "sines"=> 1.7075160000000002e-07, "sinew"=> 5.222135999999999e-07, "singe"=> 2.1927779999999998e-07, "sings"=> 3.4321259999999996e-06, "sinhs"=> 1.2728774000000003e-09, "sinks"=> 2.7771939999999997e-06, "sinky"=> 1.5349779999999998e-09, "sinus"=> 4.925356e-06, "siped"=> 9.567714e-10, "sipes"=> 4.275344e-08, "sippy"=> 9.843592000000001e-08, "sired"=> 2.50312e-07, "siree"=> 4.366456e-08, "siren"=> 2.4407299999999998e-06, "sires"=> 1.940244e-07, "sirih"=> 5.963046e-09, "siris"=> 5.847124e-08, "siroc"=> 3.173466e-09, "sirra"=> 2.2535581999999997e-08, "sirup"=> 4.169108e-08, "sisal"=> 2.3896739999999997e-07, "sises"=> 1.9667440000000002e-09, "sissy"=> 1.0932344e-06, "sista"=> 8.708128000000001e-08, "sists"=> 3.10343e-08, "sitar"=> 1.4597560000000001e-07, "sited"=> 6.986264e-07, "sites"=> 4.39733e-05, "sithe"=> 1.3241875999999999e-08, "sitka"=> 2.817216e-07, "situp"=> 1.0236337999999998e-08, "situs"=> 2.3677140000000002e-07, "siver"=> 9.989458000000001e-09, "sixer"=> 2.2686180000000003e-08, "sixes"=> 2.596431999999999e-07, "sixmo"=> 1.9648512000000004e-10, "sixte"=> 2.918834e-08, "sixth"=> 1.61036e-05, "sixty"=> 1.696034e-05, "sizar"=> 1.3356256000000001e-08, "sized"=> 1.2769219999999998e-05, "sizel"=> 8.746002e-10, "sizer"=> 9.853070000000001e-08, "sizes"=> 1.359042e-05, "skags"=> 2.8030126000000002e-09, "skail"=> 1.90754e-09, "skald"=> 6.021318e-08, "skank"=> 1.0031079999999999e-07, "skart"=> 3.869768e-09, "skate"=> 1.177418e-06, "skats"=> 6.982594e-10, "skatt"=> 2.7188198000000002e-08, "skaws"=> 5.822692000000001e-11, "skean"=> 8.952150000000001e-09, "skear"=> 2.4014802e-09, "skeds"=> 2.6420739999999998e-09, "skeed"=> 2.364696e-10, "skeef"=> 1.2235170000000001e-08, "skeen"=> 4.652552e-08, "skeer"=> 1.923993e-08, "skees"=> 1.1993174e-08, "skeet"=> 2.481204e-07, "skegg"=> 1.1200354000000001e-08, "skegs"=> 3.378838e-09, "skein"=> 3.5586520000000003e-07, "skelf"=> 2.6915176e-09, "skell"=> 2.090316e-08, "skelm"=> 7.793252e-09, "skelp"=> 1.656938e-08, "skene"=> 1.8997660000000004e-07, "skens"=> 1.6572354000000002e-10, "skeos"=> 2.0157086000000001e-10, "skeps"=> 1.1060395999999998e-08, "skers"=> 1.9590162000000002e-10, "skets"=> 4.217236e-09, "skews"=> 1.3057940000000002e-07, "skids"=> 2.783954e-07, "skied"=> 1.7800839999999998e-07, "skier"=> 3.1049899999999995e-07, "skies"=> 5.175496e-06, "skiey"=> 5.663533999999999e-09, "skiff"=> 7.425746e-07, "skill"=> 2.8420799999999997e-05, "skimo"=> 1.6156572e-09, "skimp"=> 1.212372e-07, "skims"=> 1.431978e-07, "skink"=> 1.4189592e-07, "skins"=> 4.239168e-06, "skint"=> 5.6060439999999995e-08, "skios"=> 3.9660560000000003e-10, "skips"=> 5.889976e-07, "skirl"=> 3.2256999999999996e-08, "skirr"=> 3.705532e-09, "skirt"=> 1.0001470000000002e-05, "skite"=> 7.4088240000000005e-09, "skits"=> 2.120672e-07, "skive"=> 2.168212e-08, "skivy"=> 1.1306099999999999e-10, "sklim"=> 3.71987e-10, "skoal"=> 1.714994e-08, "skody"=> 1.405356e-10, "skoff"=> 4.512849999999999e-09, "skogs"=> 3.0270540000000007e-09, "skols"=> 5.914246e-10, "skool"=> 7.922174e-08, "skort"=> 9.584896000000001e-09, "skosh"=> 4.053176000000001e-09, "skran"=> 7.067791999999999e-09, "skrik"=> 4.912163999999999e-08, "skuas"=> 4.2205240000000003e-08, "skugs"=> 1.1616349399999997e-09, "skulk"=> 1.3709919999999999e-07, "skull"=> 1.1637520000000001e-05, "skunk"=> 8.843975999999999e-07, "skyed"=> 6.2176520000000006e-09, "skyer"=> 1.4058213999999998e-08, "skyey"=> 1.2820202000000002e-08, "skyfs"=> 0.0, "skyre"=> 7.147644e-10, "skyrs"=> 7.893838e-11, "skyte"=> 3.871945800000001e-09, "slabs"=> 2.018602e-06, "slack"=> 3.938492e-06, "slade"=> 1.9670420000000003e-06, "slaes"=> 2.6811044000000003e-09, "slags"=> 1.9620119999999998e-07, "slaid"=> 2.49776676e-08, "slain"=> 4.7599119999999994e-06, "slake"=> 1.957704e-07, "slams"=> 1.124818e-06, "slane"=> 9.361334000000001e-08, "slang"=> 2.149088e-06, "slank"=> 7.987158e-09, "slant"=> 1.297766e-06, "slaps"=> 9.863598e-07, "slart"=> 1.9866857999999998e-09, "slash"=> 2.2909179999999996e-06, "slate"=> 3.560934e-06, "slats"=> 6.541836e-07, "slaty"=> 8.667516e-08, "slave"=> 2.8156920000000003e-05, "slaws"=> 9.17816e-09, "slays"=> 2.1398619999999998e-07, "slebs"=> 9.341076e-10, "sleds"=> 3.3498160000000003e-07, "sleek"=> 2.68584e-06, "sleep"=> 8.809076e-05, "sleer"=> 3.4047166e-09, "sleet"=> 6.829018e-07, "slept"=> 1.864806e-05, "slews"=> 1.551812e-08, "sleys"=> 5.0276e-10, "slice"=> 8.749728e-06, "slick"=> 4.025974000000001e-06, "slide"=> 1.49906e-05, "slier"=> 2.6254959999999996e-09, "slily"=> 4.66672e-08, "slime"=> 1.461256e-06, "slims"=> 4.585758e-08, "slimy"=> 1.247364e-06, "sling"=> 2.218954e-06, "slink"=> 3.9644879999999997e-07, "slipe"=> 2.3783379999999997e-09, "slips"=> 3.701176e-06, "slipt"=> 4.856062e-08, "slish"=> 4.166998e-09, "slits"=> 1.4427719999999998e-06, "slive"=> 1.55471e-08, "sloan"=> 2.606236e-06, "slobs"=> 5.384028e-08, "sloes"=> 3.517832e-08, "slogs"=> 1.614576e-08, "sloid"=> 4.2663124000000005e-10, "slojd"=> 5.614028400000001e-10, "slomo"=> 3.88526e-09, "sloom"=> 1.4975860000000004e-09, "sloop"=> 9.091228e-07, "sloot"=> 4.8901659999999994e-08, "slope"=> 1.636128e-05, "slops"=> 1.519476e-07, "slopy"=> 2.8197540000000004e-09, "slorm"=> 9.9597052e-10, "slosh"=> 1.7377520000000002e-07, "sloth"=> 8.071266e-07, "slots"=> 2.153256e-06, "slove"=> 3.2751988e-08, "slows"=> 1.975328e-06, "sloyd"=> 1.1431928000000001e-08, "slubb"=> 8.900332000000001e-11, "slubs"=> 4.1906579999999994e-09, "slued"=> 9.274328e-09, "slues"=> 2.658136e-09, "sluff"=> 3.3892984000000006e-08, "slugs"=> 7.319824e-07, "sluit"=> 5.3414400000000006e-08, "slump"=> 1.3717899999999999e-06, "slums"=> 1.735564e-06, "slung"=> 2.885554e-06, "slunk"=> 6.766490000000001e-07, "slurb"=> 9.687054e-10, "slurp"=> 2.607352e-07, "slurs"=> 4.527e-07, "sluse"=> 5.195228000000001e-09, "slush"=> 5.930251999999999e-07, "sluts"=> 1.924332e-07, "slyer"=> 1.0095006e-08, "slyly"=> 7.681978e-07, "slype"=> 4.785512e-09, "smaak"=> 1.649112e-08, "smack"=> 2.471332e-06, "smaik"=> 6.900258e-10, "small"=> 0.0003216436, "smalm"=> 1.3543948e-10, "smalt"=> 1.2559190000000001e-08, "smarm"=> 1.1305788e-08, "smart"=> 3.256448e-05, "smash"=> 2.7740380000000002e-06, "smaze"=> 4.5273160000000004e-10, "smear"=> 2.323496e-06, "smeek"=> 6.336326e-09, "smees"=> 1.2448848000000002e-08, "smeik"=> 1.30852e-10, "smeke"=> 5.7862162e-10, "smell"=> 2.9939439999999996e-05, "smelt"=> 1.55889e-06, "smerk"=> 1.9342579999999997e-09, "smews"=> 6.765612000000001e-10, "smile"=> 0.00010434256, "smirk"=> 3.314276e-06, "smirr"=> 2.5483979999999998e-09, "smirs"=> 3.068424e-10, "smite"=> 1.0147328e-06, "smith"=> 5.8412679999999996e-05, "smits"=> 2.58415e-07, "smock"=> 6.466382e-07, "smogs"=> 1.2040488e-08, "smoke"=> 3.28003e-05, "smoko"=> 2.0993319999999998e-08, "smoky"=> 2.6393419999999998e-06, "smolt"=> 3.650414e-08, "smoor"=> 5.0400219999999995e-09, "smoot"=> 1.98767e-07, "smore"=> 6.212556e-09, "smorg"=> 1.6833278e-09, "smote"=> 1.976328e-06, "smout"=> 6.017124e-08, "smowt"=> 8.720894000000001e-11, "smugs"=> 3.3561779999999997e-09, "smurs"=> 9.170607999999999e-11, "smush"=> 1.4079659999999999e-08, "smuts"=> 4.4019760000000007e-07, "snabs"=> 8.532578e-11, "snack"=> 3.9963860000000004e-06, "snafu"=> 7.010826e-08, "snags"=> 3.207252e-07, "snail"=> 1.7184939999999998e-06, "snake"=> 1.11802e-05, "snaky"=> 1.370934e-07, "snaps"=> 1.9743419999999998e-06, "snare"=> 1.813148e-06, "snarf"=> 1.2172066e-08, "snark"=> 2.543494e-07, "snarl"=> 1.491962e-06, "snars"=> 1.3589411999999998e-09, "snary"=> 2.176954e-09, "snash"=> 9.369562e-10, "snath"=> 3.9323360000000005e-09, "snaws"=> 9.023838e-10, "snead"=> 1.3432e-07, "sneak"=> 4.479848e-06, "sneap"=> 2.3444333999999998e-09, "snebs"=> 0.0, "sneck"=> 4.254104e-08, "sneds"=> 5.153606199999999e-10, "sneed"=> 1.4062599999999998e-07, "sneer"=> 1.9496059999999997e-06, "snees"=> 1.5829746e-10, "snell"=> 8.086502e-07, "snibs"=> 4.4296059999999997e-10, "snick"=> 1.37979e-07, "snide"=> 4.2921619999999995e-07, "snies"=> 2.929726e-09, "sniff"=> 2.112416e-06, "snift"=> 1.24386e-09, "snigs"=> 1.5560606000000003e-09, "snipe"=> 4.0851200000000005e-07, "snips"=> 1.355454e-07, "snipy"=> 1.4086846e-09, "snirt"=> 1.371316e-09, "snits"=> 4.618142e-09, "snobs"=> 2.3564940000000002e-07, "snods"=> 2.9418344e-10, "snoek"=> 6.176584000000001e-08, "snoep"=> 6.657392e-09, "snogs"=> 4.682918000000001e-09, "snoke"=> 2.3297226e-08, "snood"=> 4.193522e-08, "snook"=> 1.8633320000000002e-07, "snool"=> 9.430708e-10, "snoop"=> 4.533274e-07, "snoot"=> 3.763798e-08, "snore"=> 6.86051e-07, "snort"=> 2.0907960000000003e-06, "snots"=> 1.0586603999999999e-08, "snout"=> 1.413942e-06, "snowk"=> 3.005498e-10, "snows"=> 9.363885999999999e-07, "snowy"=> 2.64472e-06, "snubs"=> 8.000666e-08, "snuck"=> 1.7264039999999998e-06, "snuff"=> 1.418374e-06, "snugs"=> 1.2652818000000002e-08, "snush"=> 4.282132e-10, "snyes"=> 2.1687231999999997e-10, "soaks"=> 2.6938360000000003e-07, "soaps"=> 7.726154e-07, "soapy"=> 6.751977999999999e-07, "soare"=> 3.366366e-08, "soars"=> 4.37681e-07, "soave"=> 7.590568e-08, "sobas"=> 9.301054e-09, "sober"=> 5.962864e-06, "socas"=> 5.223514e-09, "soces"=> 1.0977558e-10, "socko"=> 7.2925639999999995e-09, "socks"=> 5.134772e-06, "socle"=> 4.5166599999999995e-08, "sodas"=> 4.65405e-07, "soddy"=> 7.461428e-08, "sodic"=> 1.1914756e-07, "sodom"=> 1.43773e-06, "sofar"=> 6.080446e-08, "sofas"=> 1.0249176e-06, "softa"=> 6.390439999999999e-09, "softs"=> 1.1667992e-08, "softy"=> 1.0807044e-07, "soger"=> 7.10989e-09, "soggy"=> 9.392799999999999e-07, "sohur"=> 2.3414158000000004e-10, "soils"=> 8.730498e-06, "soily"=> 2.591582e-09, "sojas"=> 2.5646559999999997e-09, "sojus"=> 7.678486000000001e-10, "sokah"=> 2.6729542e-10, "soken"=> 9.302458e-09, "sokes"=> 3.1047559999999995e-09, "sokol"=> 2.4234960000000003e-07, "solah"=> 4.882392e-09, "solan"=> 9.686802000000001e-08, "solar"=> 2.172242e-05, "solas"=> 1.671622e-07, "solde"=> 6.360511999999999e-08, "soldi"=> 8.774706e-08, "soldo"=> 2.6203719999999998e-08, "solds"=> 1.4945680000000001e-09, "soled"=> 2.9090239999999996e-07, "solei"=> 3.833079999999999e-09, "soler"=> 2.77202e-07, "soles"=> 2.0107959999999998e-06, "solid"=> 4.5094000000000006e-05, "solon"=> 6.91228e-07, "solos"=> 4.895355999999999e-07, "solum"=> 1.9891280000000002e-07, "solus"=> 1.4174440000000002e-07, "solve"=> 2.4351439999999998e-05, "soman"=> 1.1507986e-07, "somas"=> 1.87207e-08, "sonar"=> 8.630942000000001e-07, "sonce"=> 2.1478814e-09, "sonde"=> 3.478976e-08, "sones"=> 5.265602e-08, "songs"=> 2.381256e-05, "sonic"=> 2.12357e-06, "sonly"=> 1.9096879999999996e-09, "sonne"=> 1.8295866000000003e-06, "sonny"=> 2.23822e-06, "sonse"=> 5.316098e-10, "sonsy"=> 7.225408e-09, "sooey"=> 2.810848e-09, "sooks"=> 3.095526e-09, "sooky"=> 6.1384939999999996e-09, "soole"=> 5.776546e-09, "sools"=> 1.1983772e-09, "sooms"=> 3.8563879999999995e-10, "soops"=> 1.4391976000000002e-09, "soote"=> 5.544014e-09, "sooth"=> 5.169766e-07, "soots"=> 6.2288200000000004e-09, "sooty"=> 5.517996e-07, "sophs"=> 1.577529e-08, "sophy"=> 7.097214e-07, "sopor"=> 9.541456e-09, "soppy"=> 1.134796e-07, "sopra"=> 5.272676e-07, "soral"=> 9.508594e-09, "soras"=> 8.222364e-09, "sorbo"=> 1.8614759999999998e-08, "sorbs"=> 2.548534e-08, "sorda"=> 9.031718e-09, "sordo"=> 3.8640260000000004e-08, "sords"=> 8.254282e-10, "sored"=> 9.172832000000002e-09, "soree"=> 1.591722e-09, "sorel"=> 3.4617320000000005e-07, "sorer"=> 4.142484e-08, "sores"=> 1.240596e-06, "sorex"=> 4.390796e-08, "sorgo"=> 8.102296e-09, "sorns"=> 6.782576000000001e-09, "sorra"=> 2.728558e-08, "sorry"=> 7.638371999999999e-05, "sorta"=> 3.8639420000000005e-07, "sorts"=> 1.674826e-05, "sorus"=> 1.2339713999999998e-08, "soths"=> 4.484266e-10, "sotol"=> 1.6919754e-08, "souce"=> 6.8768499999999995e-09, "souct"=> 3.534644e-11, "sough"=> 5.3835339999999995e-08, "souks"=> 5.471940000000001e-08, "souls"=> 1.738568e-05, "soums"=> 4.806346e-09, "sound"=> 0.000128015, "soups"=> 1.330518e-06, "soupy"=> 1.5232359999999999e-07, "sours"=> 1.1463139999999998e-07, "souse"=> 5.4238720000000004e-08, "south"=> 0.0001485364, "souts"=> 1.5977718000000002e-10, "sowar"=> 1.2601774e-08, "sowce"=> 4.008052e-10, "sowed"=> 7.024133999999999e-07, "sower"=> 4.1379160000000004e-07, "sowff"=> 6.299898e-11, "sowfs"=> 0.0, "sowle"=> 4.4568799999999996e-08, "sowls"=> 1.0513546e-08, "sowms"=> 9.407576000000001e-11, "sownd"=> 3.459048e-09, "sowne"=> 8.44315e-09, "sowps"=> 3.1188504e-10, "sowse"=> 7.524169999999999e-10, "sowth"=> 2.4191190000000003e-09, "soyas"=> 1.3258754000000002e-09, "soyle"=> 1.666518e-08, "soyuz"=> 1.9690000000000002e-07, "sozin"=> 9.141434e-10, "space"=> 0.0001713682, "spacy"=> 2.2801380000000002e-08, "spade"=> 1.8603980000000002e-06, "spado"=> 7.68648e-09, "spaed"=> 1.4526997999999997e-09, "spaer"=> 2.218558e-09, "spaes"=> 1.4891087999999999e-09, "spags"=> 1.6339692e-09, "spahi"=> 1.7416101999999998e-08, "spail"=> 2.4650070000000004e-10, "spain"=> 2.8560019999999996e-05, "spait"=> 2.8982537000000003e-09, "spake"=> 3.342144e-06, "spald"=> 6.728950000000001e-10, "spale"=> 2.4883494e-09, "spall"=> 7.801584e-08, "spalt"=> 1.0033214e-08, "spams"=> 1.7022099999999997e-08, "spane"=> 7.997888e-09, "spang"=> 6.728774e-08, "spank"=> 6.010008e-07, "spans"=> 2.4358280000000004e-06, "spard"=> 1.1526841999999999e-09, "spare"=> 1.528534e-05, "spark"=> 8.187444e-06, "spars"=> 4.3115439999999997e-07, "spart"=> 1.5954460000000003e-08, "spasm"=> 1.732674e-06, "spate"=> 5.957708e-07, "spats"=> 1.7611460000000002e-07, "spaul"=> 4.332577999999999e-09, "spawl"=> 1.1508891999999999e-09, "spawn"=> 1.167788e-06, "spaws"=> 5.324962e-10, "spayd"=> 1.0926196000000001e-08, "spays"=> 4.695822e-09, "spaza"=> 1.9601440000000003e-08, "spazz"=> 6.755261999999999e-09, "speak"=> 0.0001067002, "speal"=> 3.732886e-09, "spean"=> 5.8548e-08, "spear"=> 6.900210000000001e-06, "speat"=> 1.0144365999999998e-09, "speck"=> 1.854064e-06, "specs"=> 8.349050000000001e-07, "spect"=> 1.0591492e-06, "speed"=> 6.271041999999999e-05, "speel"=> 7.718386e-08, "speer"=> 6.755474e-07, "speil"=> 3.0596639999999998e-09, "speir"=> 4.067992e-08, "speks"=> 5.923485000000001e-10, "speld"=> 3.36418e-09, "spelk"=> 8.382082e-10, "spell"=> 1.427038e-05, "spelt"=> 1.0245672000000001e-06, "spend"=> 4.369188e-05, "spent"=> 7.02283e-05, "speos"=> 4.485322e-09, "sperm"=> 5.756802e-06, "spets"=> 3.877742e-09, "speug"=> 1.6087728e-10, "spews"=> 8.849214e-08, "spewy"=> 1.3377584e-09, "spial"=> 7.617937999999999e-10, "spica"=> 1.300167e-07, "spice"=> 3.998987999999999e-06, "spick"=> 1.245164e-07, "spics"=> 2.121886e-08, "spicy"=> 3.1524560000000004e-06, "spide"=> 3.857332e-09, "spied"=> 2.0334079999999997e-06, "spiel"=> 3.8629119999999996e-07, "spier"=> 1.1562044e-07, "spies"=> 4.17801e-06, "spiff"=> 2.2910299999999998e-08, "spifs"=> 5.94416e-10, "spike"=> 4.41273e-06, "spiks"=> 1.5239892000000002e-09, "spiky"=> 7.528604000000001e-07, "spile"=> 3.78566e-08, "spill"=> 5.054578e-06, "spilt"=> 7.112913999999999e-07, "spims"=> 5.281782000000001e-10, "spina"=> 5.935131999999999e-07, "spine"=> 1.564424e-05, "spink"=> 1.856702e-07, "spins"=> 2.21405e-06, "spiny"=> 5.48276e-07, "spire"=> 1.14992e-06, "spirt"=> 3.510956000000001e-08, "spiry"=> 1.2340396e-08, "spite"=> 2.294582e-05, "spits"=> 8.267494e-07, "spitz"=> 4.185468e-07, "spivs"=> 1.4311980000000001e-08, "splat"=> 2.558886e-07, "splay"=> 2.030186e-07, "split"=> 2.6318160000000002e-05, "splog"=> 2.5876054e-09, "spode"=> 4.442428e-08, "spods"=> 2.2862446000000004e-09, "spoil"=> 4.2809579999999994e-06, "spoke"=> 7.67222e-05, "spoof"=> 2.8482660000000004e-07, "spook"=> 6.244061999999999e-07, "spool"=> 6.157726e-07, "spoom"=> 9.874916e-10, "spoon"=> 7.538223999999999e-06, "spoor"=> 2.3635740000000003e-07, "spoot"=> 1.0613953999999999e-09, "spore"=> 8.920885999999999e-07, "spork"=> 5.1870879999999997e-08, "sport"=> 2.5048659999999993e-05, "sposh"=> 4.463774e-10, "spots"=> 1.1913899999999998e-05, "spout"=> 9.115882e-07, "sprad"=> 2.137398e-09, "sprag"=> 1.5544192e-08, "sprat"=> 1.5375639999999999e-07, "spray"=> 9.669566e-06, "spred"=> 4.9045100000000006e-08, "spree"=> 8.952062e-07, "sprew"=> 4.4630880000000006e-10, "sprig"=> 6.240212e-07, "sprit"=> 8.069872e-08, "sprod"=> 1.0191838000000001e-08, "sprog"=> 4.040148e-08, "sprue"=> 1.616704e-07, "sprug"=> 8.117560000000001e-10, "spuds"=> 1.3972760000000002e-07, "spued"=> 4.494390000000001e-09, "spuer"=> 6.612542e-11, "spues"=> 9.494560000000001e-10, "spugs"=> 3.5876964e-10, "spule"=> 1.312288e-09, "spume"=> 9.195078000000001e-08, "spumy"=> 3.0220299999999997e-09, "spunk"=> 3.826164e-07, "spurn"=> 2.977028e-07, "spurs"=> 2.0189600000000002e-06, "spurt"=> 9.787326e-07, "sputa"=> 1.869008e-08, "spyal"=> 7.456038e-11, "spyre"=> 1.6938181199999997e-08, "squab"=> 8.292934e-08, "squad"=> 6.1114920000000005e-06, "squat"=> 2.43163e-06, "squaw"=> 5.301776e-07, "squeg"=> 1.4275744e-10, "squib"=> 1.0937346e-07, "squid"=> 1.4337419999999998e-06, "squit"=> 7.970788000000001e-09, "squiz"=> 4.957428e-09, "stabs"=> 5.335688e-07, "stack"=> 1.0189446e-05, "stade"=> 1.7794780000000004e-07, "staff"=> 7.54847e-05, "stage"=> 0.0001021426, "stags"=> 2.692662e-07, "stagy"=> 2.302338e-08, "staid"=> 8.263488e-07, "staig"=> 2.6021499999999997e-09, "stain"=> 5.291407999999999e-06, "stair"=> 2.49955e-06, "stake"=> 1.13017e-05, "stale"=> 2.71308e-06, "stalk"=> 2.482726e-06, "stall"=> 5.309028e-06, "stamp"=> 6.17104e-06, "stand"=> 9.126014e-05, "stane"=> 1.6205304000000001e-07, "stang"=> 1.4046180000000002e-07, "stank"=> 6.144347999999999e-07, "staph"=> 1.6235659999999998e-07, "staps"=> 6.514842e-09, "stare"=> 1.6138900000000002e-05, "stark"=> 8.221996000000001e-06, "starn"=> 5.6029060000000004e-08, "starr"=> 1.93241e-06, "stars"=> 3.1845639999999995e-05, "start"=> 0.000135403, "stash"=> 1.272896e-06, "state"=> 0.0004491520000000001, "stats"=> 2.365576e-06, "staun"=> 6.077082e-09, "stave"=> 1.0369296e-06, "staws"=> 2.0222623999999998e-10, "stays"=> 7.087562000000001e-06, "stead"=> 2.429294e-06, "steak"=> 4.163376e-06, "steal"=> 1.035626e-05, "steam"=> 1.670068e-05, "stean"=> 7.490208e-09, "stear"=> 1.2433046000000001e-08, "stedd"=> 3.5793882e-10, "stede"=> 8.989030000000001e-08, "steds"=> 1.2590854e-09, "steed"=> 1.476788e-06, "steek"=> 8.73432e-08, "steel"=> 3.515576e-05, "steem"=> 9.172366000000002e-09, "steen"=> 5.685872000000001e-07, "steep"=> 1.0249762e-05, "steer"=> 4.2416219999999995e-06, "steil"=> 5.56477e-08, "stein"=> 5.125034e-06, "stela"=> 2.588844e-07, "stele"=> 5.414058e-07, "stell"=> 9.703994e-08, "steme"=> 1.0810442000000001e-09, "stems"=> 7.648721999999999e-06, "stend"=> 2.9717458e-08, "steno"=> 1.526808e-07, "stens"=> 8.652050000000001e-09, "stent"=> 2.374126e-06, "steps"=> 8.093818e-05, "stept"=> 6.655630000000001e-08, "stere"=> 1.797442e-08, "stern"=> 1.2440220000000001e-05, "stets"=> 1.1745478000000001e-07, "stews"=> 5.305656e-07, "stewy"=> 8.719948000000001e-09, "steys"=> 3.0593316e-10, "stich"=> 1.9209220000000003e-07, "stick"=> 2.80941e-05, "stied"=> 1.8500281999999997e-09, "sties"=> 3.6552240000000006e-08, "stiff"=> 1.123804e-05, "stilb"=> 1.1020538e-09, "stile"=> 6.905676e-07, "still"=> 0.000522323, "stilt"=> 1.6249879999999997e-07, "stime"=> 1.0669313999999999e-08, "stims"=> 1.61418e-08, "stimy"=> 3.3759820000000005e-10, "sting"=> 4.37337e-06, "stink"=> 2.010512e-06, "stint"=> 1.3565180000000001e-06, "stipa"=> 3.6329180000000005e-08, "stipe"=> 1.648376e-07, "stire"=> 2.8978079999999997e-09, "stirk"=> 3.574339999999999e-08, "stirp"=> 5.802753999999999e-09, "stirs"=> 1.1746320000000002e-06, "stive"=> 7.37907e-09, "stivy"=> 4.198494e-10, "stoae"=> 6.245568e-10, "stoai"=> 1.5327238000000002e-09, "stoas"=> 1.785554e-08, "stoat"=> 9.5781e-08, "stobs"=> 9.815418000000001e-09, "stock"=> 5.428052e-05, "stoep"=> 8.978478e-08, "stogy"=> 8.374068e-09, "stoic"=> 2.6927499999999998e-06, "stoit"=> 3.274672e-09, "stoke"=> 9.792712000000002e-07, "stole"=> 8.510102e-06, "stoln"=> 3.89987e-09, "stoma"=> 5.3145e-07, "stomp"=> 7.949276e-07, "stond"=> 1.1516978199999998e-07, "stone"=> 6.737876e-05, "stong"=> 4.9074919999999995e-08, "stonk"=> 3.82861e-09, "stonn"=> 2.4193812e-09, "stony"=> 2.835022e-06, "stood"=> 0.0001404476, "stook"=> 1.67522e-08, "stool"=> 7.820266e-06, "stoop"=> 1.8800600000000002e-06, "stoor"=> 1.33866e-08, "stope"=> 7.089476e-08, "stops"=> 1.375304e-05, "stopt"=> 1.3081632e-07, "store"=> 5.0458480000000003e-05, "stork"=> 7.079652e-07, "storm"=> 2.8181019999999998e-05, "story"=> 0.0001636616, "stoss"=> 3.147408e-08, "stots"=> 3.716164e-09, "stott"=> 5.188124e-07, "stoun"=> 9.508534e-10, "stoup"=> 4.434668e-08, "stour"=> 1.0766029999999999e-07, "stout"=> 5.078378e-06, "stove"=> 7.315317999999999e-06, "stown"=> 5.72666e-09, "stowp"=> 7.133006e-10, "stows"=> 3.2127280000000004e-08, "strad"=> 3.0018679999999996e-08, "strae"=> 8.217678000000001e-09, "strag"=> 1.6511760000000003e-09, "strak"=> 1.0495802000000002e-08, "strap"=> 3.855976e-06, "straw"=> 9.257032e-06, "stray"=> 4.744684e-06, "strep"=> 2.8501680000000003e-07, "strew"=> 1.757176e-07, "stria"=> 1.5179060000000002e-07, "strig"=> 4.325366e-09, "strim"=> 4.586718e-09, "strip"=> 1.3579999999999999e-05, "strop"=> 9.30707e-08, "strow"=> 1.0343498e-08, "stroy"=> 1.859728e-08, "strum"=> 2.565054e-07, "strut"=> 9.760281999999999e-07, "stubs"=> 4.407776e-07, "stuck"=> 2.695448e-05, "stude"=> 1.647522e-08, "studs"=> 1.0847538e-06, "study"=> 0.00025894939999999994, "stuff"=> 3.7930639999999995e-05, "stull"=> 7.700844e-08, "stulm"=> 6.625432e-11, "stumm"=> 1.5223234e-07, "stump"=> 2.96429e-06, "stums"=> 2.3062928e-10, "stung"=> 3.442224e-06, "stunk"=> 1.6187379999999997e-07, "stuns"=> 9.536915999999998e-08, "stunt"=> 1.4894939999999997e-06, "stupa"=> 2.510988e-07, "stupe"=> 9.212208e-09, "sture"=> 6.260052e-08, "sturt"=> 2.03412e-07, "styed"=> 1.89579e-09, "styes"=> 2.0041440000000002e-08, "style"=> 6.498728000000001e-05, "styli"=> 2.1674e-08, "stylo"=> 3.797356e-08, "styme"=> 6.160667999999999e-10, "stymy"=> 4.723586e-09, "styre"=> 1.7299992e-08, "styte"=> 6.3846884e-10, "suave"=> 5.80038e-07, "subah"=> 1.8595580000000002e-08, "subas"=> 5.991371999999999e-09, "subby"=> 6.701295999999999e-09, "suber"=> 4.3846120000000006e-08, "subha"=> 3.948612e-08, "succi"=> 2.51668e-08, "sucks"=> 2.054966e-06, "sucky"=> 6.461382000000001e-08, "sucre"=> 1.7429739999999998e-07, "sudds"=> 9.521150000000001e-09, "sudor"=> 1.615636e-08, "sudsy"=> 7.453281999999999e-08, "suede"=> 5.365084e-07, "suent"=> 1.1566324e-09, "suers"=> 4.4364679999999994e-09, "suete"=> 4.516669999999999e-09, "suets"=> 4.157178e-10, "suety"=> 4.91106e-09, "sugan"=> 3.57149e-09, "sugar"=> 3.672678e-05, "sughs"=> 1.3596524e-10, "sugos"=> 1.1837592e-10, "suhur"=> 3.424054e-09, "suids"=> 1.0537168000000001e-08, "suing"=> 6.99093e-07, "suint"=> 2.811516e-09, "suite"=> 1.0663368e-05, "suits"=> 9.17598e-06, "sujee"=> 5.624096e-10, "sukhs"=> 7.041794e-11, "sukuk"=> 2.96638e-07, "sulci"=> 1.9358780000000002e-07, "sulfa"=> 1.607818e-07, "sulfo"=> 3.024406e-08, "sulks"=> 1.093924e-07, "sulky"=> 6.917608e-07, "sully"=> 1.296354e-06, "sulph"=> 9.130356e-08, "sulus"=> 5.3410340000000006e-09, "sumac"=> 3.265566e-07, "sumis"=> 1.4778380000000001e-09, "summa"=> 1.28691e-06, "sumos"=> 6.97887e-09, "sumph"=> 3.693666e-09, "sumps"=> 4.5178e-08, "sunis"=> 2.0198939999999998e-09, "sunks"=> 2.348008e-10, "sunna"=> 3.6585219999999994e-07, "sunns"=> 6.211322e-10, "sunny"=> 7.32788e-06, "sunup"=> 1.62414e-07, "super"=> 1.7772539999999996e-05, "supes"=> 2.3192259999999997e-08, "supra"=> 8.738288e-06, "surah"=> 3.736904e-07, "sural"=> 1.770266e-07, "suras"=> 8.641783999999999e-08, "surat"=> 4.940761999999999e-07, "surds"=> 2.167174e-08, "sured"=> 7.621002000000001e-08, "surer"=> 4.531796e-07, "sures"=> 1.537016e-07, "surfs"=> 4.6554379999999997e-08, "surfy"=> 3.5922580000000004e-09, "surge"=> 6.605296e-06, "surgy"=> 3.8872860000000005e-09, "surly"=> 1.021374e-06, "surra"=> 2.3698359999999997e-08, "sused"=> 7.486246000000001e-10, "suses"=> 6.759664e-10, "sushi"=> 1.1941241999999998e-06, "susus"=> 3.0844999999999994e-09, "sutor"=> 2.060428e-08, "sutra"=> 8.760086000000001e-07, "sutta"=> 2.362488e-07, "swabs"=> 4.953752e-07, "swack"=> 6.548406e-09, "swads"=> 7.382659999999999e-10, "swage"=> 2.219354e-08, "swags"=> 8.051286000000001e-08, "swail"=> 7.953922000000001e-09, "swain"=> 1.06721e-06, "swale"=> 1.423916e-07, "swaly"=> 1.5037696e-10, "swami"=> 1.4129859999999999e-06, "swamp"=> 3.858654e-06, "swamy"=> 1.446262e-07, "swang"=> 2.28734e-08, "swank"=> 2.375434e-07, "swans"=> 1.1461720000000002e-06, "swaps"=> 1.1879628000000003e-06, "swapt"=> 9.555836e-10, "sward"=> 2.8950620000000004e-07, "sware"=> 2.3205359999999997e-07, "swarf"=> 2.4386020000000004e-08, "swarm"=> 3.819324e-06, "swart"=> 3.590454e-07, "swash"=> 1.4065020000000002e-07, "swath"=> 8.008481999999999e-07, "swats"=> 1.487504e-07, "swayl"=> 1.1900999999999999e-10, "sways"=> 3.833434e-07, "sweal"=> 1.0661044e-09, "swear"=> 1.18474e-05, "sweat"=> 1.619272e-05, "swede"=> 6.313223999999999e-07, "sweed"=> 5.600431999999999e-09, "sweel"=> 5.982058000000001e-10, "sweep"=> 7.5364180000000005e-06, "sweer"=> 1.957258e-08, "swees"=> 9.303216e-10, "sweet"=> 5.17679e-05, "sweir"=> 5.683844e-09, "swell"=> 4.82468e-06, "swelt"=> 2.405426e-09, "swept"=> 1.661408e-05, "swerf"=> 6.652702e-10, "sweys"=> 1.2070624e-10, "swies"=> 3.0615258e-10, "swift"=> 1.182028e-05, "swigs"=> 9.167796e-08, "swile"=> 1.6014449999999997e-09, "swill"=> 1.934302e-07, "swims"=> 7.042224000000001e-07, "swine"=> 2.45332e-06, "swing"=> 1.194188e-05, "swink"=> 3.016878e-08, "swipe"=> 1.583616e-06, "swire"=> 5.254936e-08, "swirl"=> 2.0801139999999997e-06, "swish"=> 9.396918000000001e-07, "swiss"=> 8.182583999999998e-06, "swith"=> 6.6173160000000004e-09, "swits"=> 1.597258e-09, "swive"=> 9.380814e-09, "swizz"=> 1.0030332000000001e-08, "swobs"=> 4.6126100000000003e-10, "swole"=> 1.294336e-08, "swoln"=> 2.29543e-08, "swoon"=> 8.022820000000001e-07, "swoop"=> 1.1176800000000002e-06, "swops"=> 2.38477e-09, "swopt"=> 2.3156966e-10, "sword"=> 3.392964e-05, "swore"=> 6.732671999999999e-06, "sworn"=> 5.780768e-06, "swots"=> 1.269488e-08, "swoun"=> 7.903724000000001e-10, "swung"=> 1.4250660000000003e-05, "sybbe"=> 3.0355140000000003e-10, "sybil"=> 1.3443320000000002e-06, "syboe"=> 5.4402020000000004e-11, "sybow"=> 1.4708094e-10, "sycee"=> 3.986744000000001e-09, "syces"=> 8.756354e-09, "sycon"=> 4.227448000000001e-09, "syens"=> 2.353624e-10, "syker"=> 1.2999864000000003e-09, "sykes"=> 1.3172139999999998e-06, "sylis"=> 5.375850999999999e-09, "sylph"=> 1.452634e-07, "sylva"=> 2.3175299999999997e-07, "symar"=> 7.021712e-10, "synch"=> 2.049406e-07, "syncs"=> 5.054852e-08, "synds"=> 9.953948399999999e-10, "syned"=> 6.155478e-10, "synes"=> 3.221136e-08, "synod"=> 1.596896e-06, "synth"=> 3.836158e-07, "syped"=> 0.0, "sypes"=> 2.6761618e-10, "syphs"=> 7.692472e-11, "syrah"=> 1.871272e-07, "syren"=> 6.141504000000001e-08, "syrup"=> 4.9829120000000005e-06, "sysop"=> 5.758888e-09, "sythe"=> 9.287588e-09, "syver"=> 7.247874000000001e-10, "taals"=> 1.4326486000000002e-09, "taata"=> 5.815533999999999e-09, "tabby"=> 8.702556e-07, "taber"=> 2.813152e-07, "tabes"=> 5.293055999999999e-08, "tabid"=> 2.673274e-08, "tabis"=> 2.14428e-09, "tabla"=> 1.513006e-07, "table"=> 0.0002835378, "taboo"=> 2.648868e-06, "tabor"=> 7.323369999999999e-07, "tabun"=> 5.280484e-08, "tabus"=> 2.066616e-08, "tacan"=> 2.3919520000000004e-08, "taces"=> 3.768698e-09, "tacet"=> 1.920546e-08, "tache"=> 6.634926e-08, "tacho"=> 1.845134e-08, "tachs"=> 5.063235000000001e-09, "tacit"=> 3.298844e-06, "tacks"=> 3.741246e-07, "tacky"=> 5.53949e-07, "tacos"=> 9.071206e-07, "tacts"=> 1.644246e-08, "taels"=> 2.913228e-07, "taffy"=> 4.816434e-07, "tafia"=> 1.0870699999999999e-08, "taggy"=> 1.3825992e-09, "tagma"=> 1.0808316e-08, "tahas"=> 1.0277726e-09, "tahrs"=> 2.6515892e-09, "taiga"=> 2.367986e-07, "taigs"=> 2.600936e-09, "taiko"=> 1.1894438000000001e-07, "tails"=> 4.5141320000000006e-06, "tains"=> 8.29986e-08, "taint"=> 1.1914700000000002e-06, "taira"=> 1.35521e-07, "taish"=> 4.1911100000000004e-09, "taits"=> 5.52726e-09, "tajes"=> 1.818538e-09, "takas"=> 1.106977e-08, "taken"=> 0.00020062919999999998, "taker"=> 1.180268e-06, "takes"=> 9.187058e-05, "takhi"=> 1.1758447999999999e-08, "takin"=> 5.342821999999999e-07, "takis"=> 8.685466e-08, "takky"=> 4.7096584e-10, "talak"=> 1.3711468e-08, "talaq"=> 9.951715999999999e-08, "talar"=> 2.600738e-07, "talas"=> 7.591646e-08, "talcs"=> 5.138429999999999e-09, "talcy"=> 1.087894e-09, "talea"=> 2.119521e-08, "taler"=> 1.4637715999999999e-07, "tales"=> 1.4208059999999998e-05, "talks"=> 1.3454999999999999e-05, "talky"=> 4.57447e-08, "talls"=> 1.3569938e-08, "tally"=> 1.8653e-06, "talma"=> 8.870622e-08, "talon"=> 1.1569264e-06, "talpa"=> 2.342626e-08, "taluk"=> 5.0688960000000005e-08, "talus"=> 5.508086e-07, "tamal"=> 2.27721e-08, "tamed"=> 1.203112e-06, "tamer"=> 3.646246e-07, "tames"=> 8.551182e-08, "tamin"=> 2.9258e-08, "tamis"=> 8.562969999999999e-08, "tammy"=> 1.7490639999999998e-06, "tamps"=> 1.782186e-08, "tanas"=> 6.448332000000001e-09, "tanga"=> 1.0986885999999998e-07, "tangi"=> 6.755286e-08, "tango"=> 1.3290700000000001e-06, "tangs"=> 4.454578e-08, "tangy"=> 5.526956e-07, "tanhs"=> 1.55002e-10, "tanka"=> 1.1736096000000001e-07, "tanks"=> 9.852162e-06, "tanky"=> 6.34696e-09, "tanna"=> 1.6279168e-07, "tansy"=> 3.862738e-07, "tanti"=> 7.42151e-08, "tanto"=> 9.161178000000001e-07, "tanty"=> 1.0752296000000001e-08, "tapas"=> 6.549823999999999e-07, "taped"=> 1.8690680000000002e-06, "tapen"=> 1.9269212e-09, "taper"=> 1.4024260000000002e-06, "tapes"=> 2.544974e-06, "tapet"=> 1.92334e-09, "tapir"=> 1.3233748e-07, "tapis"=> 5.640023999999999e-08, "tappa"=> 6.448638e-08, "tapus"=> 4.9486518000000004e-08, "taras"=> 2.6063279999999996e-07, "tardo"=> 4.1994700000000003e-08, "tardy"=> 5.698166e-07, "tared"=> 2.399456e-08, "tares"=> 3.4444480000000003e-07, "targa"=> 7.079316e-08, "targe"=> 3.71801e-08, "tarns"=> 4.376814e-08, "taroc"=> 2.1276676e-09, "tarok"=> 1.05773e-08, "taros"=> 1.38952e-08, "tarot"=> 1.0218518e-06, "tarps"=> 1.926446e-07, "tarre"=> 7.46168e-09, "tarry"=> 9.374283999999999e-07, "tarsi"=> 1.3556148e-07, "tarts"=> 7.004648e-07, "tarty"=> 2.3885760000000003e-08, "tasar"=> 2.1781876000000003e-08, "tased"=> 2.3122860000000003e-08, "taser"=> 3.949152e-07, "tases"=> 5.153667999999999e-09, "tasks"=> 3.6863479999999996e-05, "tassa"=> 3.39813e-08, "tasse"=> 5.55825e-08, "tasso"=> 4.806018000000001e-07, "taste"=> 3.957334e-05, "tasty"=> 2.6166380000000003e-06, "tatar"=> 5.665602e-07, "tater"=> 1.661924e-07, "tates"=> 1.09504e-07, "taths"=> 2.471264e-10, "tatie"=> 9.796533999999999e-09, "tatou"=> 5.6924738000000004e-08, "tatts"=> 2.0078360000000002e-08, "tatty"=> 2.5648060000000004e-07, "tatus"=> 2.3959840000000003e-08, "taube"=> 1.7896600000000002e-07, "tauld"=> 2.101482e-08, "taunt"=> 1.019362e-06, "tauon"=> 5.4916179999999995e-09, "taupe"=> 1.1305245999999999e-07, "tauts"=> 8.372861999999999e-10, "tavah"=> 1.2048576e-09, "tavas"=> 7.895756e-09, "taver"=> 1.9773520000000002e-09, "tawai"=> 4.19414e-09, "tawas"=> 3.6446268e-08, "tawed"=> 4.127526e-09, "tawer"=> 1.0837116e-09, "tawie"=> 7.259810000000001e-10, "tawny"=> 1.1370922e-06, "tawse"=> 2.3413240000000003e-08, "tawts"=> 1.5634747999999998e-10, "taxed"=> 2.8363419999999997e-06, "taxer"=> 6.785474000000001e-09, "taxes"=> 2.379906e-05, "taxis"=> 1.541918e-06, "taxol"=> 1.6948419999999997e-07, "taxon"=> 5.61783e-07, "taxor"=> 7.321653999999999e-10, "taxus"=> 1.1683980000000001e-07, "tayra"=> 3.440574e-09, "tazza"=> 2.456884e-08, "tazze"=> 8.618863999999999e-09, "teach"=> 3.4867959999999996e-05, "teade"=> 5.34412e-10, "teads"=> 1.05388e-09, "teaed"=> 2.931172e-10, "teaks"=> 2.634162e-09, "teals"=> 2.389546e-08, "teams"=> 2.4886479999999995e-05, "tears"=> 5.589652e-05, "teary"=> 5.305766000000001e-07, "tease"=> 3.637382e-06, "teats"=> 2.1117539999999998e-07, "teaze"=> 1.3764146e-08, "techs"=> 4.969752000000001e-07, "techy"=> 3.3128759999999995e-08, "tecta"=> 2.7294859999999996e-08, "teddy"=> 5.228998e-06, "teels"=> 2.639318e-09, "teems"=> 1.291852e-07, "teend"=> 3.480924e-10, "teene"=> 4.462564e-09, "teens"=> 5.554182e-06, "teeny"=> 4.0367380000000003e-07, "teers"=> 9.010824e-09, "teeth"=> 4.272086e-05, "teffs"=> 1.7510380000000002e-09, "teggs"=> 8.881196e-10, "tegua"=> 8.780325999999999e-10, "tegus"=> 5.803586e-09, "tehrs"=> 4.455634e-11, "teiid"=> 1.4613942e-09, "teils"=> 1.6235348e-08, "teind"=> 7.053982000000001e-09, "teins"=> 2.58678e-08, "telae"=> 1.2930012e-09, "telco"=> 9.648292e-08, "teles"=> 8.287324e-08, "telex"=> 2.8062600000000004e-07, "telia"=> 3.6079140000000007e-08, "telic"=> 1.616442e-07, "tells"=> 3.5332319999999995e-05, "telly"=> 5.224188000000001e-07, "teloi"=> 9.160928e-09, "telos"=> 7.519258e-07, "temed"=> 1.5870244e-08, "temes"=> 2.4972220000000004e-08, "tempi"=> 1.560274e-07, "tempo"=> 3.609972e-06, "temps"=> 1.5987960000000002e-06, "tempt"=> 2.026768e-06, "temse"=> 2.6116540000000002e-09, "tench"=> 1.8909739999999998e-07, "tends"=> 1.566944e-05, "tendu"=> 3.443656e-08, "tenes"=> 1.9082556e-08, "tenet"=> 1.27876e-06, "tenge"=> 2.899398e-08, "tenia"=> 4.036224e-08, "tenne"=> 8.139174e-08, "tenno"=> 3.331908e-08, "tenny"=> 5.649502000000001e-08, "tenon"=> 1.809032e-07, "tenor"=> 2.498178e-06, "tense"=> 1.2569379999999999e-05, "tenth"=> 8.562568e-06, "tents"=> 5.4223699999999996e-06, "tenty"=> 2.9941080000000003e-09, "tenue"=> 5.0844599999999995e-08, "tepal"=> 7.923714e-09, "tepas"=> 6.670204e-09, "tepee"=> 2.116346e-07, "tepid"=> 6.916036e-07, "tepoy"=> 1.5402622e-10, "terai"=> 1.1428198e-07, "teras"=> 2.7518760000000006e-08, "terce"=> 4.781628e-08, "terek"=> 1.7282469999999998e-07, "teres"=> 3.3161439999999996e-07, "terfe"=> 3.5224179999999997e-10, "terfs"=> 4.3213486e-09, "terga"=> 4.085498e-08, "terms"=> 0.0001709524, "terne"=> 1.3330214000000001e-08, "terns"=> 2.5823100000000003e-07, "terra"=> 3.2458360000000005e-06, "terry"=> 8.632788e-06, "terse"=> 9.162343999999999e-07, "terts"=> 4.3646e-09, "tesla"=> 1.4944699999999999e-06, "testa"=> 3.81264e-07, "teste"=> 9.489966e-08, "tests"=> 4.70585e-05, "testy"=> 3.2354500000000003e-07, "tetes"=> 9.897276e-09, "teths"=> 2.332567e-10, "tetra"=> 4.44825e-07, "tetri"=> 7.831874e-09, "teuch"=> 2.912428e-09, "teugh"=> 1.0526016000000002e-09, "tewed"=> 3.792554e-09, "tewel"=> 8.955886000000001e-10, "tewit"=> 9.966947999999999e-10, "texas"=> 2.6460500000000002e-05, "texes"=> 1.5492548e-08, "texts"=> 4.141962e-05, "thack"=> 1.8363282e-08, "thagi"=> 1.9594599999999998e-09, "thaim"=> 2.0770618000000002e-08, "thale"=> 3.9899859999999995e-08, "thali"=> 5.373647999999999e-08, "thana"=> 1.3525256000000002e-07, "thane"=> 7.10703e-07, "thang"=> 2.752274e-07, "thank"=> 8.176628000000001e-05, "thans"=> 1.3985678e-08, "thanx"=> 1.1839761999999999e-08, "tharm"=> 1.7919498e-09, "thars"=> 7.096964e-09, "thaws"=> 1.115064e-07, "thawy"=> 6.096572e-10, "thebe"=> 3.845572e-08, "theca"=> 1.245668e-07, "theed"=> 1.2409334e-08, "theek"=> 1.6207319999999997e-08, "thees"=> 4.998515999999999e-08, "theft"=> 8.024104e-06, "thegn"=> 4.253252e-08, "theic"=> 7.17253e-10, "thein"=> 1.496754e-07, "their"=> 0.002028576, "thelf"=> 7.497138000000001e-10, "thema"=> 1.512966e-07, "theme"=> 2.572092e-05, "thens"=> 2.4015559999999997e-08, "theow"=> 1.8579906e-09, "there"=> 0.001763988, "therm"=> 1.793658e-07, "these"=> 0.0010919380000000002, "thesp"=> 2.280622e-09, "theta"=> 1.208508e-06, "thete"=> 2.550096e-09, "thews"=> 5.6486399999999996e-08, "thewy"=> 8.609442e-10, "thick"=> 4.485326e-05, "thief"=> 7.071489999999999e-06, "thigh"=> 8.832278e-06, "thigs"=> 3.627294e-09, "thilk"=> 2.2374985999999997e-09, "thill"=> 4.70088e-08, "thine"=> 6.66412e-06, "thing"=> 0.0002504406, "think"=> 0.0004720592, "thins"=> 2.074534e-07, "thiol"=> 6.052604000000001e-07, "third"=> 0.00015541219999999997, "thirl"=> 3.784165999999999e-09, "thoft"=> 7.073853999999999e-09, "thole"=> 5.186734e-08, "tholi"=> 1.5198274e-09, "thong"=> 1.1015204e-06, "thorn"=> 3.4461520000000005e-06, "thoro"=> 2.2066316e-08, "thorp"=> 2.8624579999999997e-07, "those"=> 0.0006227976, "thous"=> 3.036002e-08, "thowl"=> 6.548177999999999e-10, "thrae"=> 4.4851371e-09, "thraw"=> 9.718052000000001e-09, "three"=> 0.0004516528, "threw"=> 3.346528e-05, "thrid"=> 1.1723222e-08, "thrip"=> 1.2009431999999999e-08, "throb"=> 1.4914e-06, "throe"=> 4.7029199999999995e-08, "throw"=> 2.839144e-05, "thrum"=> 3.3452739999999996e-07, "thuds"=> 3.122272e-07, "thugs"=> 1.54873e-06, "thuja"=> 6.828867999999999e-08, "thumb"=> 1.4745620000000001e-05, "thump"=> 2.498024e-06, "thunk"=> 3.8421120000000004e-07, "thurl"=> 6.824144e-09, "thuya"=> 1.219101e-08, "thyme"=> 2.429484e-06, "thymi"=> 4.482826e-09, "thymy"=> 7.462894e-09, "tians"=> 2.8682060000000004e-08, "tiara"=> 6.193882e-07, "tiars"=> 9.037124e-11, "tibia"=> 1.400848e-06, "tical"=> 8.186911999999999e-08, "ticca"=> 5.034006e-09, "ticed"=> 1.8975600000000002e-08, "tices"=> 5.267246e-08, "tichy"=> 7.819939999999999e-08, "ticks"=> 1.59968e-06, "ticky"=> 2.944426e-08, "tidal"=> 4.210322e-06, "tiddy"=> 2.5138999999999998e-08, "tided"=> 2.87151e-08, "tides"=> 2.507114e-06, "tiers"=> 1.524716e-06, "tiffs"=> 4.217499999999999e-08, "tifos"=> 9.3883802e-10, "tifts"=> 7.74223e-10, "tiger"=> 8.578304e-06, "tiges"=> 4.4407660000000005e-09, "tight"=> 3.513514e-05, "tigon"=> 1.1999648e-08, "tikas"=> 4.541366e-09, "tikes"=> 7.756014000000001e-09, "tikis"=> 8.863146e-09, "tikka"=> 1.2868320000000003e-07, "tilak"=> 2.60261e-07, "tilde"=> 1.198664e-07, "tiled"=> 1.57146e-06, "tiler"=> 3.6557780000000005e-08, "tiles"=> 4.1624120000000005e-06, "tills"=> 1.482756e-07, "tilly"=> 2.3076219999999995e-06, "tilth"=> 6.026138e-08, "tilts"=> 9.101788000000001e-07, "timbo"=> 4.065722e-08, "timed"=> 2.736962e-06, "timer"=> 2.6775080000000004e-06, "times"=> 0.0002415618, "timid"=> 3.230534e-06, "timon"=> 5.81878e-07, "timps"=> 3.493414e-08, "tinas"=> 5.018926e-09, "tinct"=> 3.624116e-08, "tinds"=> 3.8754500000000003e-10, "tinea"=> 4.80105e-07, "tined"=> 3.037424e-08, "tines"=> 2.4917300000000003e-07, "tinge"=> 1.4399839999999997e-06, "tings"=> 7.567804e-08, "tinks"=> 2.408812e-08, "tinny"=> 3.6919280000000003e-07, "tints"=> 6.865389999999999e-07, "tinty"=> 4.4548584000000006e-10, "tipis"=> 6.708746e-08, "tippy"=> 2.673954e-07, "tipsy"=> 7.881790000000001e-07, "tired"=> 3.4687120000000003e-05, "tires"=> 3.880706e-06, "tirls"=> 1.074159e-09, "tiros"=> 3.626764e-08, "tirrs"=> 3.2027326e-10, "titan"=> 1.562548e-06, "titch"=> 8.497166e-08, "titer"=> 4.6983539999999997e-07, "tithe"=> 9.816842e-07, "titis"=> 1.271702e-08, "title"=> 5.519962e-05, "titre"=> 1.988498e-07, "titty"=> 1.0065265999999999e-07, "titup"=> 1.96677614e-09, "tiyin"=> 8.25954e-10, "tiyns"=> 0.0, "tizes"=> 1.1700806e-09, "tizzy"=> 1.5176180000000002e-07, "toads"=> 6.522368e-07, "toady"=> 1.1270122000000002e-07, "toast"=> 6.962177999999999e-06, "toaze"=> 6.234138000000001e-10, "tocks"=> 1.8277639999999998e-08, "tocky"=> 1.2112282e-09, "tocos"=> 1.2045714000000001e-09, "today"=> 0.0001486988, "todde"=> 4.2617559999999995e-09, "toddy"=> 2.7723379999999997e-07, "toeas"=> 1.8914524e-10, "toffs"=> 5.686242e-08, "toffy"=> 2.010422e-08, "tofts"=> 3.4144080000000004e-08, "tofus"=> 1.6489606e-09, "togae"=> 6.3967799999999995e-09, "togas"=> 9.296351999999999e-08, "toged"=> 9.190349999999999e-10, "toges"=> 4.87992e-10, "togue"=> 4.749664e-09, "tohos"=> 8.408418e-11, "toile"=> 7.100022e-08, "toils"=> 5.94693e-07, "toing"=> 2.837052e-08, "toise"=> 1.168049e-08, "toits"=> 2.1460760000000002e-08, "tokay"=> 6.318800000000001e-08, "toked"=> 9.883894e-09, "token"=> 6.557176e-06, "toker"=> 6.442706e-08, "tokes"=> 3.053928e-08, "tokos"=> 8.030346000000001e-09, "tolan"=> 1.5772940000000002e-07, "tolar"=> 2.98942e-08, "tolas"=> 1.2868232000000001e-08, "toled"=> 3.756506e-09, "toles"=> 1.882482e-08, "tolls"=> 8.38028e-07, "tolly"=> 1.3831219999999999e-07, "tolts"=> 4.7165359999999995e-09, "tolus"=> 8.250166000000001e-10, "tolyl"=> 3.385658e-08, "toman"=> 1.2264148000000001e-07, "tombs"=> 3.5815280000000003e-06, "tomes"=> 4.4398959999999997e-07, "tomia"=> 3.2422079999999995e-09, "tommy"=> 9.613466e-06, "tomos"=> 5.035446e-08, "tonal"=> 1.538918e-06, "tondi"=> 1.9318259999999998e-08, "tondo"=> 8.054658000000001e-08, "toned"=> 1.70933e-06, "toner"=> 4.027408e-07, "tones"=> 8.054008e-06, "toney"=> 1.1676123999999999e-07, "tonga"=> 7.365697999999999e-07, "tongs"=> 9.740254e-07, "tonic"=> 2.942766e-06, "tonka"=> 7.891031999999999e-08, "tonks"=> 1.1578201999999998e-07, "tonne"=> 6.264574e-07, "tonus"=> 7.74076e-08, "tools"=> 5.505788e-05, "tooms"=> 1.447479e-08, "toons"=> 2.6016660000000005e-08, "tooth"=> 9.639326e-06, "toots"=> 3.6408860000000003e-07, "topaz"=> 6.358438e-07, "toped"=> 3.1827199999999995e-09, "topee"=> 1.85993e-08, "topek"=> 2.2267214000000004e-09, "toper"=> 4.444336e-08, "topes"=> 2.7150480000000002e-08, "tophe"=> 6.64687e-10, "tophi"=> 4.398942e-08, "tophs"=> 1.0053096e-10, "topic"=> 3.46705e-05, "topis"=> 6.25877e-09, "topoi"=> 3.347872e-07, "topos"=> 6.791084e-07, "toppy"=> 3.71226e-08, "toque"=> 1.391026e-07, "torah"=> 4.371028e-06, "toran"=> 9.961056000000001e-08, "toras"=> 5.521021999999999e-09, "torch"=> 5.4154560000000006e-06, "torcs"=> 1.9890780000000002e-08, "tores"=> 1.4522486e-08, "toric"=> 1.8026019999999999e-07, "torii"=> 1.0249206e-07, "toros"=> 9.671359999999999e-08, "torot"=> 2.812264e-09, "torrs"=> 6.750192000000001e-09, "torse"=> 1.4426176e-08, "torsi"=> 8.385362e-09, "torsk"=> 5.262474000000001e-09, "torso"=> 4.155722e-06, "torta"=> 8.061348e-08, "torte"=> 1.411608e-07, "torts"=> 1.1386638000000002e-06, "torus"=> 6.53751e-07, "tosas"=> 1.0393434e-09, "tosed"=> 4.78298e-10, "toses"=> 1.4322486000000003e-09, "toshy"=> 1.4758188e-09, "tossy"=> 3.0354999999999997e-09, "total"=> 0.0001316048, "toted"=> 1.0286952e-07, "totem"=> 1.0440734e-06, "toter"=> 1.1816104000000001e-08, "totes"=> 1.249128e-07, "totty"=> 9.662054e-08, "touch"=> 6.440013999999999e-05, "tough"=> 1.736112e-05, "touks"=> 7.158224e-11, "touns"=> 1.4401598e-08, "tours"=> 7.491460000000001e-06, "touse"=> 2.9358859999999997e-08, "tousy"=> 1.383509e-09, "touts"=> 1.986452e-07, "touze"=> 5.786158e-09, "touzy"=> 1.9555702e-10, "towed"=> 1.083712e-06, "towel"=> 9.591914e-06, "tower"=> 2.331996e-05, "towie"=> 1.7721465999999997e-08, "towns"=> 1.99373e-05, "towny"=> 5.954153999999999e-09, "towse"=> 4.066346e-08, "towsy"=> 3.182286e-09, "towts"=> 7.970316e-11, "towze"=> 2.00636e-10, "towzy"=> 1.1047233999999999e-10, "toxic"=> 1.2547939999999998e-05, "toxin"=> 3.05779e-06, "toyed"=> 9.402286e-07, "toyer"=> 2.594814e-09, "toyon"=> 1.5246592e-08, "toyos"=> 1.2435248e-09, "tozed"=> 5.801642e-11, "tozes"=> 3.433914e-11, "tozie"=> 4.346144e-10, "trabs"=> 8.522112000000001e-09, "trace"=> 2.248894e-05, "track"=> 3.963176e-05, "tract"=> 1.322114e-05, "trade"=> 0.00011082429999999999, "trads"=> 8.987422e-09, "tragi"=> 8.705676000000001e-08, "traik"=> 1.12236162e-09, "trail"=> 2.7501180000000002e-05, "train"=> 4.924028e-05, "trait"=> 7.849066000000001e-06, "tramp"=> 2.223082e-06, "trams"=> 5.116484e-07, "trank"=> 4.440102e-08, "tranq"=> 4.489834e-08, "trans"=> 3.050178e-05, "trant"=> 1.0232406000000002e-07, "trape"=> 9.031762e-09, "traps"=> 4.41065e-06, "trapt"=> 2.9502400000000004e-09, "trash"=> 6.82154e-06, "trass"=> 8.400232e-09, "trats"=> 4.962113999999999e-10, "tratt"=> 4.970732e-09, "trave"=> 3.4755320000000005e-08, "trawl"=> 3.70317e-07, "trayf"=> 1.78278e-09, "trays"=> 2.1705020000000003e-06, "tread"=> 3.6799060000000003e-06, "treat"=> 3.3986900000000005e-05, "treck"=> 2.8301760000000002e-09, "treed"=> 1.3256204e-07, "treen"=> 2.9298859999999998e-08, "trees"=> 6.438368e-05, "trefa"=> 2.1234105999999996e-09, "treif"=> 7.135564e-09, "treks"=> 2.789724e-07, "trema"=> 2.209896e-08, "trems"=> 2.6588822e-09, "trend"=> 2.063926e-05, "tress"=> 2.3001040000000003e-07, "trest"=> 1.6645674000000004e-08, "trets"=> 2.9180259999999997e-09, "trews"=> 6.321094e-08, "treyf"=> 6.3432660000000005e-09, "treys"=> 9.88316e-09, "triac"=> 4.3785279999999995e-08, "triad"=> 2.3843480000000002e-06, "trial"=> 5.735798e-05, "tribe"=> 1.4016540000000001e-05, "trice"=> 2.999868e-07, "trick"=> 1.219864e-05, "tride"=> 5.029775999999999e-09, "tried"=> 0.00012798000000000003, "trier"=> 8.739134e-07, "tries"=> 1.37384e-05, "triff"=> 5.7208600000000004e-09, "trigo"=> 1.3427336e-07, "trigs"=> 2.973562e-09, "trike"=> 1.0789502e-07, "trild"=> 3.8267839999999997e-10, "trill"=> 4.074302e-07, "trims"=> 2.388188e-07, "trine"=> 1.51612e-07, "trins"=> 1.4480786e-09, "triol"=> 3.092224e-08, "trior"=> 1.6389514e-09, "trios"=> 1.802726e-07, "tripe"=> 3.019496e-07, "trips"=> 1.0877153999999999e-05, "tripy"=> 1.5509088e-10, "trist"=> 2.107494e-07, "trite"=> 5.52296e-07, "troad"=> 5.8251159999999995e-08, "troak"=> 2.9992594000000003e-10, "troat"=> 2.7497999999999997e-09, "trock"=> 1.0981054e-08, "trode"=> 5.521152e-08, "trods"=> 9.190359999999999e-09, "trogs"=> 2.86828e-08, "trois"=> 9.243303999999999e-07, "troke"=> 1.3423604000000003e-08, "troll"=> 1.4581420000000002e-06, "tromp"=> 1.955322e-07, "trona"=> 3.4742280000000004e-08, "tronc"=> 2.5464300000000005e-08, "trone"=> 5.761032000000001e-08, "tronk"=> 1.4302879999999999e-08, "trons"=> 2.22315e-08, "troop"=> 4.3155419999999995e-06, "trooz"=> 6.966938000000001e-10, "trope"=> 2.159768e-06, "troth"=> 3.9523220000000007e-07, "trots"=> 2.320096e-07, "trout"=> 4.609505999999999e-06, "trove"=> 7.44246e-07, "trows"=> 1.3025562e-08, "troys"=> 8.979328e-09, "truce"=> 2.366012e-06, "truck"=> 2.649334e-05, "trued"=> 1.241312e-08, "truer"=> 1.0729198e-06, "trues"=> 1.565202e-08, "trugo"=> 9.577050000000001e-09, "trugs"=> 4.1791660000000005e-09, "trull"=> 7.059168e-08, "truly"=> 4.902888e-05, "trump"=> 1.3776704e-05, "trunk"=> 1.271696e-05, "truss"=> 1.0275496e-06, "trust"=> 9.09199e-05, "truth"=> 0.0001228962, "tryer"=> 4.504788e-09, "tryke"=> 8.566212000000001e-10, "tryma"=> 3.4626e-10, "tryps"=> 8.987198e-10, "tryst"=> 4.522726e-07, "tsade"=> 8.38395e-09, "tsadi"=> 1.4649388e-09, "tsars"=> 1.79631e-07, "tsked"=> 1.798524e-07, "tsuba"=> 8.344587999999999e-09, "tsubo"=> 2.19484e-08, "tuans"=> 1.890466e-09, "tuart"=> 1.677856e-08, "tuath"=> 2.3922319999999997e-08, "tubae"=> 7.817618e-09, "tubal"=> 6.46965e-07, "tubar"=> 1.3273952e-09, "tubas"=> 3.256324e-08, "tubby"=> 2.65303e-07, "tubed"=> 3.97388e-08, "tuber"=> 5.014984e-07, "tubes"=> 7.813902e-06, "tucks"=> 4.683598e-07, "tufas"=> 6.622732e-09, "tuffe"=> 8.5365e-10, "tuffs"=> 7.37427e-08, "tufts"=> 1.2577100000000001e-06, "tufty"=> 6.365102e-08, "tugra"=> 2.765322e-09, "tuile"=> 1.4330847999999999e-08, "tuina"=> 1.7474572e-08, "tuism"=> 2.3254499999999997e-09, "tuktu"=> 1.5788676e-09, "tules"=> 2.0940579999999997e-08, "tulip"=> 8.27482e-07, "tulle"=> 3.017154e-07, "tulpa"=> 1.8403792e-08, "tulsi"=> 1.681302e-07, "tumid"=> 3.4438319999999997e-08, "tummy"=> 1.462222e-06, "tumor"=> 2.1188919999999998e-05, "tumps"=> 2.547122e-09, "tumpy"=> 1.6465824e-09, "tunas"=> 8.43315e-08, "tunds"=> 2.2843380000000003e-10, "tuned"=> 3.909986e-06, "tuner"=> 2.009686e-07, "tunes"=> 2.466726e-06, "tungs"=> 2.0728500000000003e-09, "tunic"=> 2.68746e-06, "tunny"=> 9.132387999999999e-08, "tupek"=> 1.2103312e-09, "tupik"=> 3.6068620000000003e-09, "tuple"=> 1.0528866e-06, "tuque"=> 1.836772e-08, "turbo"=> 8.272364e-07, "turds"=> 9.905496e-08, "turfs"=> 5.0486859999999996e-08, "turfy"=> 2.4453259999999998e-08, "turks"=> 4.7871860000000006e-06, "turme"=> 2.7378627999999995e-09, "turms"=> 8.235061999999999e-09, "turns"=> 4.0022020000000005e-05, "turnt"=> 1.9845459999999998e-08, "turps"=> 2.018646e-08, "turrs"=> 3.0430419999999997e-10, "tushy"=> 8.010976e-09, "tusks"=> 6.938638e-07, "tusky"=> 9.820655999999999e-09, "tutee"=> 4.1552e-08, "tutor"=> 3.952922e-06, "tutti"=> 8.13761e-07, "tutty"=> 1.915164e-08, "tutus"=> 5.9247140000000006e-08, "tuxes"=> 4.745078e-08, "tuyer"=> 3.525317e-10, "twaes"=> 1.39304e-10, "twain"=> 3.2280200000000004e-06, "twals"=> 5.030694e-11, "twang"=> 4.4653960000000006e-07, "twank"=> 6.706642e-10, "twats"=> 2.8498240000000002e-08, "tways"=> 3.3577442200000002e-09, "tweak"=> 7.329592e-07, "tweed"=> 1.3122560000000002e-06, "tweel"=> 4.8752640000000004e-08, "tween"=> 7.120977999999999e-07, "tweep"=> 3.5855319999999995e-09, "tweer"=> 7.439702e-10, "tweet"=> 1.7037800000000002e-06, "twerk"=> 2.284226e-08, "twerp"=> 7.605532e-08, "twice"=> 3.7745419999999994e-05, "twier"=> 6.731476000000001e-10, "twigs"=> 2.050514e-06, "twill"=> 6.678156e-07, "twilt"=> 2.3791439999999997e-09, "twine"=> 1.080312e-06, "twink"=> 6.469372e-08, "twins"=> 9.639563999999998e-06, "twiny"=> 8.858418e-10, "twire"=> 2.0061196e-09, "twirl"=> 5.34403e-07, "twirp"=> 2.4666256e-09, "twist"=> 8.993386e-06, "twite"=> 1.613304e-08, "twits"=> 3.49107e-08, "twixt"=> 3.5866100000000004e-07, "twoer"=> 4.0295540000000006e-10, "twyer"=> 3.0926479999999995e-10, "tyees"=> 7.210714e-10, "tyers"=> 5.791052e-08, "tying"=> 3.5428139999999997e-06, "tyiyn"=> 4.560786e-10, "tykes"=> 3.8640700000000005e-08, "tyler"=> 8.954939999999999e-06, "tymps"=> 1.9339382e-10, "tynde"=> 1.2502925999999998e-09, "tyned"=> 5.12267e-10, "tynes"=> 3.670048e-08, "typal"=> 6.278312e-09, "typed"=> 3.6545480000000004e-06, "types"=> 9.750486e-05, "typey"=> 1.1569196e-09, "typic"=> 7.340762e-08, "typos"=> 2.2219300000000004e-07, "typps"=> 9.487656e-11, "typto"=> 1.1948872e-10, "tyran"=> 6.322628e-08, "tyred"=> 1.919788e-08, "tyres"=> 1.0079334000000002e-06, "tyros"=> 3.644652e-08, "tythe"=> 1.1423248e-08, "tzars"=> 5.8994360000000004e-09, "udals"=> 2.2099862000000002e-10, "udder"=> 2.6351539999999996e-07, "udons"=> 9.065106000000001e-11, "ugali"=> 2.338806e-08, "ugged"=> 1.8056279999999999e-09, "uhlan"=> 3.627524e-08, "uhuru"=> 1.3269260000000001e-07, "ukase"=> 3.7577060000000005e-08, "ulama"=> 6.516284e-07, "ulans"=> 1.3999056e-09, "ulcer"=> 2.7188719999999996e-06, "ulema"=> 2.6160199999999996e-07, "ulmin"=> 5.476684e-10, "ulnad"=> 3.6619586e-10, "ulnae"=> 1.2458878e-08, "ulnar"=> 1.572132e-06, "ulnas"=> 3.790484e-09, "ulpan"=> 7.947011999999999e-09, "ultra"=> 5.0995940000000005e-06, "ulvas"=> 6.93402e-10, "ulyie"=> 1.705987e-10, "ulzie"=> 3.9504936e-10, "umami"=> 2.69216e-07, "umbel"=> 4.307436e-08, "umber"=> 2.4901639999999995e-07, "umble"=> 8.734006e-08, "umbos"=> 2.5163598e-09, "umbra"=> 2.241022e-07, "umbre"=> 5.752168e-09, "umiac"=> 6.91744e-11, "umiak"=> 1.3097738e-08, "umiaq"=> 3.87347e-09, "ummah"=> 3.151304e-07, "ummas"=> 5.013746e-09, "ummed"=> 6.798781999999999e-09, "umped"=> 2.529688e-09, "umphs"=> 1.3151468e-09, "umpie"=> 4.752093400000001e-10, "umpty"=> 1.150634e-08, "umrah"=> 4.51925e-08, "umras"=> 2.3043364e-10, "unais"=> 1.3058842000000001e-09, "unapt"=> 2.183572e-08, "unarm"=> 1.1479592e-08, "unary"=> 2.96363e-07, "unaus"=> 2.0199080000000001e-10, "unbag"=> 5.977908e-10, "unban"=> 2.930154e-09, "unbar"=> 3.1907499999999996e-08, "unbed"=> 2.5244820000000004e-10, "unbid"=> 1.3300238e-08, "unbox"=> 9.456532e-09, "uncap"=> 2.18827e-08, "unces"=> 8.618808000000001e-10, "uncia"=> 1.8397776000000002e-08, "uncle"=> 3.954616e-05, "uncos"=> 1.0440434e-09, "uncoy"=> 9.113665999999999e-11, "uncus"=> 5.0346799999999997e-08, "uncut"=> 4.904904e-07, "undam"=> 5.922158e-09, "undee"=> 1.2915382e-09, "under"=> 0.000455683, "undid"=> 1.362908e-06, "undos"=> 1.9818899999999997e-09, "undue"=> 3.022344e-06, "undug"=> 4.520469999999999e-09, "uneth"=> 5.790346e-10, "unfed"=> 7.367205999999999e-08, "unfit"=> 2.294658e-06, "unfix"=> 1.711434e-08, "ungag"=> 2.179144e-09, "unget"=> 2.337402e-09, "ungod"=> 2.2262700000000004e-09, "ungot"=> 7.191668e-10, "ungum"=> 6.220073999999999e-10, "unhat"=> 5.458832e-10, "unhip"=> 1.057679e-08, "unica"=> 6.623203999999999e-08, "unify"=> 1.158548e-06, "union"=> 8.600657999999999e-05, "unite"=> 4.5379440000000006e-06, "units"=> 4.837218e-05, "unity"=> 2.377636e-05, "unjam"=> 6.096798e-09, "unked"=> 2.2283168000000002e-09, "unket"=> 3.965422e-10, "unkid"=> 4.865877e-10, "unlaw"=> 4.571731999999999e-09, "unlay"=> 3.7165679999999995e-09, "unled"=> 4.10061e-09, "unlet"=> 9.225447999999999e-09, "unlid"=> 1.1543354000000002e-09, "unlit"=> 5.621308e-07, "unman"=> 4.4814100000000006e-08, "unmet"=> 9.501952e-07, "unmew"=> 4.8170024e-10, "unmix"=> 8.578647999999999e-09, "unpay"=> 1.0951789999999998e-09, "unpeg"=> 3.98684e-09, "unpen"=> 1.2376202000000002e-09, "unpin"=> 4.70195e-08, "unred"=> 7.712273999999999e-10, "unrid"=> 1.1550524e-10, "unrig"=> 2.39849e-09, "unrip"=> 2.749942e-09, "unsaw"=> 2.0572828e-10, "unsay"=> 1.1349332e-07, "unsee"=> 5.7689039999999994e-08, "unset"=> 8.093514000000001e-08, "unsew"=> 1.1501734e-09, "unsex"=> 1.8500719999999998e-08, "unsod"=> 1.0797028e-10, "untax"=> 1.60018e-09, "untie"=> 8.381442000000001e-07, "until"=> 0.0003000674, "untin"=> 2.537936e-09, "unwed"=> 3.274464e-07, "unwet"=> 3.3660020000000004e-09, "unwit"=> 2.436326e-09, "unwon"=> 5.68414e-09, "unzip"=> 3.581486e-07, "upbow"=> 1.6471903999999999e-09, "upbye"=> 3.383446e-10, "updos"=> 7.588616000000001e-09, "updry"=> 1.1651376000000002e-10, "upend"=> 1.672864e-07, "upjet"=> 4.991822e-11, "uplay"=> 1.1924492e-09, "upled"=> 5.904522e-10, "uplit"=> 4.2542260000000005e-09, "upped"=> 4.156412e-07, "upper"=> 5.8731919999999994e-05, "upran"=> 1.0170224e-10, "uprun"=> 1.1264194e-10, "upsee"=> 1.283617e-09, "upset"=> 1.929382e-05, "upsey"=> 3.8021060000000006e-10, "uptak"=> 1.5056692000000002e-09, "upter"=> 3.2511406e-10, "uptie"=> 2.4420082000000003e-10, "uraei"=> 1.1043474e-08, "urali"=> 2.0998688e-09, "uraos"=> 1.0928076000000001e-10, "urare"=> 4.191778e-10, "urari"=> 9.584986e-10, "urase"=> 1.6911492e-09, "urate"=> 2.440732e-07, "urban"=> 6.701002e-05, "urbex"=> 4.936822e-09, "urbia"=> 1.6638148000000002e-09, "urdee"=> 1.009306e-10, "ureal"=> 2.1579467999999998e-09, "ureas"=> 3.2328480000000004e-08, "uredo"=> 9.28089e-09, "ureic"=> 6.677614e-10, "urena"=> 1.89039e-08, "urent"=> 2.77835138e-08, "urged"=> 1.2529340000000002e-05, "urger"=> 6.666322e-09, "urges"=> 3.4105280000000006e-06, "urial"=> 1.0275518e-08, "urine"=> 1.0791088e-05, "urite"=> 1.9854878e-09, "urman"=> 3.1978320000000006e-08, "urnal"=> 5.671234e-09, "urned"=> 1.2595640000000002e-08, "urped"=> 6.235658e-10, "ursae"=> 1.5158732e-08, "ursid"=> 2.4138940000000003e-09, "urson"=> 2.052325e-08, "urubu"=> 3.596304e-09, "urvas"=> 1.3654056e-10, "usage"=> 1.727268e-05, "users"=> 4.194264e-05, "usher"=> 2.0070299999999996e-06, "using"=> 0.00029952, "usnea"=> 2.204168e-08, "usque"=> 2.863836e-07, "usual"=> 4.76482e-05, "usure"=> 1.4284364e-08, "usurp"=> 5.600184e-07, "usury"=> 8.578446e-07, "uteri"=> 1.230102e-07, "utile"=> 1.524752e-07, "utter"=> 9.776142000000001e-06, "uveal"=> 1.743132e-07, "uveas"=> 1.4892852e-10, "uvula"=> 1.768496e-07, "vacua"=> 4.73146e-08, "vaded"=> 3.2999059999999996e-09, "vades"=> 3.870946e-09, "vagal"=> 6.414347999999999e-07, "vague"=> 1.2424460000000002e-05, "vagus"=> 7.67141e-07, "vails"=> 1.878148e-08, "vaire"=> 1.2137296e-08, "vairs"=> 2.0589344e-09, "vairy"=> 6.1019159999999995e-09, "vakas"=> 1.3333451999999999e-09, "vakil"=> 6.46627e-08, "vales"=> 2.261442e-07, "valet"=> 2.005048e-06, "valid"=> 2.209962e-05, "valis"=> 9.363665999999999e-08, "valor"=> 1.5496039999999999e-06, "valse"=> 1.0211318e-07, "value"=> 0.0002267028, "valve"=> 1.185168e-05, "vamps"=> 2.9409920000000003e-07, "vampy"=> 1.411174e-08, "vanda"=> 1.49015e-07, "vaned"=> 1.5884019999999998e-08, "vanes"=> 3.134194e-07, "vangs"=> 3.6415819999999996e-09, "vants"=> 3.3030160000000003e-08, "vaped"=> 7.0055779999999995e-09, "vaper"=> 4.28954e-09, "vapes"=> 6.225746e-09, "vapid"=> 2.5142399999999996e-07, "vapor"=> 6.693542000000001e-06, "varan"=> 4.40182e-08, "varas"=> 9.227406e-08, "vardy"=> 8.74847e-08, "varec"=> 1.8212320000000001e-09, "vares"=> 1.4664960000000001e-08, "varia"=> 2.45139e-07, "varix"=> 4.782582e-08, "varna"=> 3.1128539999999997e-07, "varus"=> 5.25554e-07, "varve"=> 1.1992134000000002e-08, "vasal"=> 1.8653694e-08, "vases"=> 1.4602639999999999e-06, "vasts"=> 2.9124419999999997e-09, "vasty"=> 3.150768e-08, "vatic"=> 2.724028e-08, "vatus"=> 1.102221e-09, "vauch"=> 4.1003866000000004e-10, "vault"=> 4.019464000000001e-06, "vaunt"=> 1.1260456000000001e-07, "vaute"=> 1.1347972000000002e-09, "vauts"=> 3.6240484e-10, "vawte"=> 5.3400906e-10, "vaxes"=> 6.72501e-10, "veale"=> 1.0082854000000001e-07, "veals"=> 1.0920665999999998e-08, "vealy"=> 4.960614e-10, "veena"=> 2.1492359999999998e-07, "veeps"=> 2.626322e-09, "veers"=> 2.8235960000000003e-07, "veery"=> 1.6861232e-08, "vegan"=> 2.26023e-06, "vegas"=> 4.702126e-06, "veges"=> 4.3921239999999995e-09, "vegie"=> 1.584336e-08, "vegos"=> 4.665675999999999e-10, "vehme"=> 1.7131954000000002e-09, "veils"=> 1.0740099999999999e-06, "veily"=> 2.660382e-10, "veins"=> 1.092232e-05, "veiny"=> 6.802108000000001e-08, "velar"=> 2.181896e-07, "velds"=> 9.323564000000001e-09, "veldt"=> 1.43508e-07, "veles"=> 5.974064e-08, "vells"=> 4.04954e-09, "velum"=> 1.300768e-07, "venae"=> 6.463454e-08, "venal"=> 2.8524460000000004e-07, "vends"=> 1.6680936000000002e-08, "vendu"=> 2.936424e-08, "veney"=> 8.002748e-09, "venge"=> 3.946216e-08, "venin"=> 1.7245586e-08, "venom"=> 2.417992e-06, "vents"=> 1.240532e-06, "venue"=> 5.218342e-06, "venus"=> 5.683246000000001e-06, "verbs"=> 7.988855999999999e-06, "verge"=> 4.4605159999999995e-06, "verra"=> 2.190424e-07, "verry"=> 7.872706000000001e-08, "verse"=> 2.33729e-05, "verso"=> 2.7973960000000004e-06, "verst"=> 3.856982e-08, "verts"=> 9.053836e-08, "vertu"=> 1.4619980000000001e-07, "verve"=> 3.29142e-07, "vespa"=> 1.999678e-07, "vesta"=> 5.712474e-07, "vests"=> 8.804501999999999e-07, "vetch"=> 2.725896e-07, "vexed"=> 2.119064e-06, "vexer"=> 4.075968000000001e-09, "vexes"=> 9.630362e-08, "vexil"=> 3.93932e-10, "vezir"=> 2.4248399999999998e-08, "vials"=> 1.0355960000000002e-06, "viand"=> 1.84907e-08, "vibes"=> 5.236002e-07, "vibex"=> 1.6542733e-09, "vibey"=> 2.621528e-09, "vicar"=> 3.050388e-06, "viced"=> 2.95126e-09, "vices"=> 2.875816e-06, "vichy"=> 1.091499e-06, "video"=> 4.036538e-05, "viers"=> 1.1900434000000002e-08, "views"=> 5.090146e-05, "viewy"=> 1.2958242e-09, "vifda"=> 3.1781900000000004e-10, "viffs"=> 0.0, "vigas"=> 1.4602780000000001e-08, "vigia"=> 1.1928246e-08, "vigil"=> 1.527096e-06, "vigor"=> 2.356196e-06, "vilde"=> 2.7651806e-07, "viler"=> 5.1013860000000004e-08, "villa"=> 7.399578e-06, "villi"=> 4.5210919999999995e-07, "vills"=> 1.7941240000000002e-08, "vimen"=> 1.2019132e-09, "vinal"=> 1.3498173999999999e-08, "vinas"=> 1.839942e-08, "vinca"=> 1.2780599999999997e-07, "vined"=> 8.94225e-09, "viner"=> 2.836498e-07, "vines"=> 4.074296e-06, "vinew"=> 6.786076000000001e-11, "vinic"=> 3.3294264e-09, "vinos"=> 2.505032e-08, "vints"=> 7.8378e-10, "vinyl"=> 2.79678e-06, "viola"=> 2.068956e-06, "viold"=> 2.068863e-10, "viols"=> 8.046218e-08, "viper"=> 1.05921e-06, "viral"=> 9.458854000000001e-06, "vired"=> 3.7892139999999995e-10, "vireo"=> 7.869294000000001e-08, "vires"=> 2.9762639999999996e-07, "virga"=> 5.057256e-08, "virge"=> 3.37892e-08, "virid"=> 3.803448e-09, "virls"=> 1.984041e-10, "virtu"=> 5.8336840000000003e-08, "virus"=> 1.9826960000000002e-05, "visas"=> 1.5357739999999999e-06, "vised"=> 4.277608e-08, "vises"=> 2.994238e-08, "visie"=> 7.634812e-09, "visit"=> 6.77293e-05, "visne"=> 1.6138638e-08, "vison"=> 4.481938e-08, "visor"=> 8.742812e-07, "vista"=> 2.357458e-06, "visto"=> 3.102466e-07, "vitae"=> 1.5645464e-06, "vital"=> 2.489664e-05, "vitas"=> 5.52187e-08, "vitex"=> 5.709472e-08, "vitro"=> 8.49345e-06, "vitta"=> 1.4143712000000002e-08, "vivas"=> 1.0718082e-07, "vivat"=> 4.710699999999999e-08, "vivda"=> 1.0491344e-10, "viver"=> 3.3551919999999996e-08, "vives"=> 2.874334e-07, "vivid"=> 8.11974e-06, "vixen"=> 5.342600000000001e-07, "vizir"=> 7.710922e-08, "vizor"=> 3.178752e-08, "vleis"=> 1.7317300000000002e-08, "vlies"=> 1.2416408e-08, "vlogs"=> 7.316686e-08, "voars"=> 3.6154342e-10, "vocab"=> 1.0071549999999999e-07, "vocal"=> 8.715668e-06, "voces"=> 1.4868424e-07, "voddy"=> 1.5079590000000001e-09, "vodka"=> 2.8804e-06, "vodou"=> 2.414586e-07, "vodun"=> 4.381976e-08, "voema"=> 0.0, "vogie"=> 7.640882400000001e-10, "vogue"=> 1.8532540000000001e-06, "voice"=> 0.00020778960000000001, "voids"=> 1.200984e-06, "voila"=> 1.9934239999999995e-07, "voile"=> 1.1502579999999999e-07, "voips"=> 4.53503e-10, "volae"=> 2.606954e-10, "volar"=> 4.0153860000000004e-07, "voled"=> 5.138208e-10, "voles"=> 2.798202e-07, "volet"=> 3.64865e-08, "volks"=> 1.8939386e-07, "volta"=> 6.469455999999999e-07, "volte"=> 2.838554e-07, "volti"=> 1.901734e-08, "volts"=> 1.1733774e-06, "volva"=> 2.328632e-08, "volve"=> 1.3639417999999998e-08, "vomer"=> 8.694e-08, "vomit"=> 2.3257659999999997e-06, "voted"=> 6.932230000000001e-06, "voter"=> 4.116912e-06, "votes"=> 1.0162998000000002e-05, "vouch"=> 6.864891999999999e-07, "vouge"=> 6.453492e-10, "voulu"=> 8.752808e-08, "vowed"=> 3.32822e-06, "vowel"=> 3.6803739999999997e-06, "vower"=> 2.5953416e-09, "voxel"=> 7.260246e-07, "vozhd"=> 1.4626101999999998e-08, "vraic"=> 1.8273256e-09, "vrils"=> 2.4617883999999997e-10, "vroom"=> 2.0006240000000003e-07, "vrous"=> 1.4759212e-10, "vrouw"=> 2.5348058e-07, "vrows"=> 1.2352832e-09, "vuggs"=> 1.5931226e-10, "vuggy"=> 1.877166e-08, "vughs"=> 4.3041916e-09, "vughy"=> 8.907278000000001e-10, "vulgo"=> 5.145982000000001e-08, "vulns"=> 1.5363295999999999e-09, "vulva"=> 7.497834e-07, "vutty"=> 0.0, "vying"=> 6.945894e-07, "waacs"=> 1.5179734000000002e-08, "wacke"=> 7.005335999999999e-09, "wacko"=> 7.835851999999999e-08, "wacks"=> 3.725976e-08, "wacky"=> 3.55536e-07, "wadds"=> 1.0675928e-09, "waddy"=> 6.972284000000001e-08, "waded"=> 1.0994639999999999e-06, "wader"=> 5.650908e-08, "wades"=> 1.0131286000000001e-07, "wadge"=> 1.3719320000000001e-08, "wadis"=> 9.646834e-08, "wadts"=> 8.64787e-11, "wafer"=> 1.648104e-06, "waffs"=> 3.6229791999999997e-10, "wafts"=> 2.099884e-07, "waged"=> 2.098268e-06, "wager"=> 1.9750319999999998e-06, "wages"=> 1.795236e-05, "wagga"=> 1.2000201999999998e-07, "wagon"=> 9.47814e-06, "wagyu"=> 4.417278e-08, "wahoo"=> 9.354073999999999e-08, "waide"=> 1.2558066e-08, "waifs"=> 1.43963e-07, "waift"=> 2.1348325999999996e-10, "wails"=> 5.593553999999999e-07, "wains"=> 3.497776e-08, "wairs"=> 2.780512e-10, "waist"=> 1.9007760000000002e-05, "waite"=> 6.763323999999999e-07, "waits"=> 3.823794e-06, "waive"=> 1.104494e-06, "wakas"=> 2.5560896e-08, "waked"=> 6.502975999999999e-07, "waken"=> 4.093978e-07, "waker"=> 3.27367e-08, "wakes"=> 2.507638e-06, "wakfs"=> 4.818816e-09, "waldo"=> 1.203244e-06, "walds"=> 1.3399945999999999e-09, "waled"=> 1.2360314e-08, "waler"=> 1.9157860000000004e-08, "wales"=> 1.380946e-05, "walie"=> 1.817216e-09, "walis"=> 1.59047e-08, "walks"=> 1.519718e-05, "walla"=> 3.5516119999999997e-07, "walls"=> 5.35034e-05, "wally"=> 1.52695e-06, "walty"=> 2.4152940000000003e-09, "waltz"=> 2.0002379999999997e-06, "wamed"=> 6.096764e-10, "wames"=> 1.77596e-09, "wamus"=> 5.217098e-10, "wands"=> 4.773928e-07, "waned"=> 1.1746819999999999e-06, "wanes"=> 2.706402e-07, "waney"=> 3.2530279999999997e-09, "wangs"=> 2.470246e-08, "wanks"=> 5.701516e-09, "wanky"=> 8.919945999999999e-09, "wanle"=> 5.396524e-10, "wanly"=> 1.8614299999999997e-07, "wanna"=> 4.114206e-06, "wants"=> 5.699364e-05, "wanty"=> 4.221348e-09, "wanze"=> 1.532578e-09, "waqfs"=> 5.338952e-08, "warbs"=> 3.3728328e-10, "warby"=> 5.311512e-08, "wards"=> 2.744124e-06, "wared"=> 2.630014e-09, "wares"=> 1.9563960000000003e-06, "warez"=> 1.360187e-08, "warks"=> 8.52072e-09, "warms"=> 8.593618000000001e-07, "warns"=> 2.797302e-06, "warps"=> 2.1208259999999999e-07, "warre"=> 2.6482339999999996e-07, "warst"=> 3.8911000000000003e-08, "warts"=> 9.55002e-07, "warty"=> 1.612814e-07, "wases"=> 1.9433086e-09, "washy"=> 1.3842899999999998e-07, "wasms"=> 1.1340642e-09, "wasps"=> 1.0184398e-06, "waspy"=> 2.083644e-08, "waste"=> 4.196058e-05, "wasts"=> 1.6827680000000002e-09, "watap"=> 8.871682000000001e-10, "watch"=> 7.34427e-05, "water"=> 0.0003303872, "watts"=> 3.8402760000000005e-06, "wauff"=> 5.592678e-11, "waugh"=> 8.534228000000001e-07, "wauks"=> 3.5242035999999996e-10, "waulk"=> 1.3864537999999999e-09, "wauls"=> 3.1731142000000003e-10, "waurs"=> 7.608528e-11, "waved"=> 1.8426799999999998e-05, "waver"=> 9.41037e-07, "waves"=> 3.29438e-05, "wavey"=> 8.250137999999999e-09, "wawas"=> 3.6551440000000003e-09, "wawes"=> 3.3259139999999998e-09, "wawls"=> 1.7537325999999998e-10, "waxed"=> 1.423814e-06, "waxen"=> 2.88969e-07, "waxer"=> 1.855182e-08, "waxes"=> 4.6383960000000003e-07, "wayed"=> 8.235134e-09, "wazir"=> 2.027558e-07, "wazoo"=> 2.374222e-08, "weald"=> 2.233974e-07, "weals"=> 4.7394979999999995e-08, "weamb"=> 6.975324e-11, "weans"=> 4.524352e-08, "wears"=> 4.924811999999999e-06, "weary"=> 9.744464e-06, "weave"=> 3.1284339999999996e-06, "webby"=> 4.9966199999999996e-08, "weber"=> 8.792123999999999e-06, "wecht"=> 2.0490699999999998e-08, "wedel"=> 1.330598e-07, "wedge"=> 4.073570000000001e-06, "wedgy"=> 2.91388e-09, "weeds"=> 4.847118000000001e-06, "weedy"=> 4.1800739999999996e-07, "weeke"=> 5.203082e-08, "weeks"=> 7.89023e-05, "weels"=> 1.2449404000000001e-08, "weems"=> 2.981546e-07, "weens"=> 2.6797426000000002e-08, "weeny"=> 6.312017999999999e-08, "weeps"=> 5.80758e-07, "weepy"=> 2.049754e-07, "weest"=> 2.05091856e-08, "weete"=> 3.2365240000000004e-09, "weets"=> 4.81133e-09, "wefte"=> 1.2680696e-10, "wefts"=> 3.263566e-08, "weids"=> 1.8660719999999998e-10, "weigh"=> 5.571156e-06, "weils"=> 5.684541999999999e-09, "weird"=> 1.3784579999999999e-05, "weirs"=> 1.9880760000000002e-07, "weise"=> 2.7653919999999997e-07, "weize"=> 3.950265999999999e-09, "wekas"=> 1.2738352000000001e-09, "welch"=> 2.204274e-06, "welds"=> 5.675012e-07, "welke"=> 1.9734088e-07, "welks"=> 9.817368e-09, "welkt"=> 4.6420081999999994e-10, "wells"=> 1.1860439999999999e-05, "welly"=> 5.623521999999999e-08, "welsh"=> 6.172553999999999e-06, "welts"=> 3.4812120000000003e-07, "wembs"=> 5.0291840000000004e-11, "wench"=> 9.202224e-07, "wends"=> 1.0616979999999999e-07, "wenge"=> 1.895868e-08, "wenny"=> 6.6202479999999995e-09, "wents"=> 1.9337286e-09, "weros"=> 1.7668376e-10, "wersh"=> 2.614824e-09, "wests"=> 6.588014e-08, "wetas"=> 2.815244e-09, "wetly"=> 1.4313740000000001e-07, "wexed"=> 2.099448e-09, "wexes"=> 4.972736e-10, "whack"=> 1.0250464000000001e-06, "whale"=> 5.57469e-06, "whamo"=> 1.488164e-09, "whams"=> 6.423799999999999e-09, "whang"=> 8.999115999999999e-08, "whaps"=> 2.752624e-09, "whare"=> 1.5892394000000002e-07, "wharf"=> 2.249484e-06, "whata"=> 2.7164480000000002e-08, "whats"=> 2.3012680000000002e-07, "whaup"=> 5.8501540000000006e-09, "whaur"=> 6.534102e-08, "wheal"=> 1.296246e-07, "whear"=> 1.6849764e-08, "wheat"=> 1.415516e-05, "wheel"=> 2.178336e-05, "wheen"=> 6.11638e-08, "wheep"=> 7.017954e-09, "wheft"=> 3.185393e-10, "whelk"=> 6.103636e-08, "whelm"=> 3.963754e-08, "whelp"=> 2.163258e-07, "whens"=> 3.1286859999999996e-08, "where"=> 0.000765294, "whets"=> 4.246044e-08, "whews"=> 3.8524000000000006e-10, "wheys"=> 2.0195319999999998e-09, "which"=> 0.002061152, "whids"=> 1.8515054000000001e-09, "whiff"=> 1.5275779999999998e-06, "whift"=> 5.664952e-10, "whigs"=> 1.1682062e-06, "while"=> 0.0005801339999999999, "whilk"=> 5.7955520000000005e-08, "whims"=> 1.0442952e-06, "whine"=> 1.607964e-06, "whins"=> 1.4952846e-08, "whiny"=> 2.750848e-07, "whios"=> 1.6514257999999998e-10, "whips"=> 1.3262340000000002e-06, "whipt"=> 4.7812139999999997e-08, "whirl"=> 1.5420540000000002e-06, "whirr"=> 2.2853180000000001e-07, "whirs"=> 3.733736e-08, "whish"=> 5.7977440000000005e-08, "whisk"=> 3.091706e-06, "whiss"=> 1.9613000000000002e-09, "whist"=> 4.4711280000000003e-07, "white"=> 0.0002339336, "whits"=> 1.0027859999999999e-08, "whity"=> 1.2172054e-08, "whizz"=> 2.4192480000000003e-07, "whole"=> 0.00021619120000000003, "whomp"=> 4.2767180000000006e-08, "whoof"=> 2.088572e-08, "whoop"=> 8.112173999999999e-07, "whoot"=> 1.0374706000000001e-08, "whops"=> 1.6463839999999999e-09, "whore"=> 3.207802e-06, "whorl"=> 2.85498e-07, "whort"=> 1.434083e-09, "whose"=> 0.0001381142, "whoso"=> 4.811414e-07, "whows"=> 2.5770870000000003e-10, "whump"=> 9.084386e-08, "whups"=> 2.900316e-09, "whyda"=> 1.058738e-10, "wicca"=> 2.1224259999999996e-07, "wicks"=> 4.915838e-07, "wicky"=> 1.216632e-08, "widdy"=> 2.1228826e-08, "widen"=> 2.4311580000000002e-06, "wider"=> 2.595518e-05, "wides"=> 1.693868e-08, "widow"=> 1.0801408e-05, "width"=> 1.4595999999999999e-05, "wield"=> 1.7214480000000002e-06, "wiels"=> 3.2609019999999996e-09, "wifed"=> 3.470718e-09, "wifes"=> 2.2991539999999997e-08, "wifey"=> 8.144898000000001e-08, "wifie"=> 1.9202379999999998e-08, "wifty"=> 9.20323e-10, "wigan"=> 2.818798e-07, "wigga"=> 1.9797588000000002e-09, "wiggy"=> 2.8926479999999998e-08, "wight"=> 1.0150764e-06, "wikis"=> 3.47415e-07, "wilco"=> 5.532338000000001e-08, "wilds"=> 8.517724000000001e-07, "wiled"=> 1.688672e-08, "wiles"=> 8.466404e-07, "wilga"=> 1.1086288e-08, "wilis"=> 1.2515619999999999e-08, "wilja"=> 1.8348424e-09, "wills"=> 4.93886e-06, "willy"=> 2.3649799999999997e-06, "wilts"=> 1.6953380000000003e-07, "wimps"=> 1.1465686e-07, "wimpy"=> 1.933354e-07, "wince"=> 1.387536e-06, "winch"=> 8.013344e-07, "winds"=> 1.105694e-05, "windy"=> 1.947308e-06, "wined"=> 9.160944000000002e-08, "wines"=> 4.765896e-06, "winey"=> 3.411746e-08, "winge"=> 5.0722920000000005e-08, "wings"=> 2.0217559999999998e-05, "wingy"=> 7.328108000000001e-09, "winks"=> 9.206265999999998e-07, "winna"=> 7.818584000000001e-08, "winns"=> 4.255056e-09, "winos"=> 4.985348e-08, "winze"=> 4.768528e-09, "wiped"=> 1.262422e-05, "wiper"=> 2.88338e-07, "wipes"=> 1.431708e-06, "wired"=> 3.566484e-06, "wirer"=> 1.6162254e-09, "wires"=> 6.110545999999999e-06, "wirra"=> 1.0799168e-08, "wised"=> 5.928656e-08, "wiser"=> 3.4541919999999997e-06, "wises"=> 1.6876260000000002e-08, "wisha"=> 9.293084e-09, "wisht"=> 1.0832192000000001e-07, "wisps"=> 9.489764e-07, "wispy"=> 6.422342e-07, "wists"=> 2.0399362000000003e-09, "witan"=> 5.9102039999999995e-08, "witch"=> 1.094272e-05, "wited"=> 3.6346099999999996e-10, "wites"=> 1.9343659999999998e-09, "withe"=> 4.3259179999999995e-08, "withs"=> 1.890768e-08, "withy"=> 5.146134e-08, "witty"=> 2.28608e-06, "wived"=> 7.736884e-09, "wiver"=> 2.8129674e-09, "wives"=> 1.453242e-05, "wizen"=> 1.982758e-08, "wizes"=> 5.9761686e-10, "woads"=> 9.667314e-10, "woald"=> 7.293496e-10, "wocks"=> 5.874504e-10, "wodge"=> 1.8017019999999998e-08, "woful"=> 8.247194e-08, "wojus"=> 3.3491064e-10, "woken"=> 2.7415699999999997e-06, "woker"=> 2.9112540000000003e-09, "wokka"=> 1.2128863999999998e-09, "wolds"=> 7.206468000000001e-08, "wolfs"=> 6.163408000000001e-08, "wolly"=> 1.4236328e-08, "wolve"=> 3.052374e-09, "woman"=> 0.00024400340000000002, "wombs"=> 2.43212e-07, "womby"=> 1.2824396e-09, "women"=> 0.000325652, "womyn"=> 3.809234e-08, "wonga"=> 2.65629e-08, "wongi"=> 1.955198e-09, "wonks"=> 3.830158e-08, "wonky"=> 2.0761479999999999e-07, "wonts"=> 3.81063e-09, "woods"=> 2.4509440000000002e-05, "woody"=> 2.984542e-06, "wooed"=> 4.204638e-07, "wooer"=> 9.326172e-08, "woofs"=> 1.735794e-08, "woofy"=> 4.40446e-09, "woold"=> 6.020288e-09, "wools"=> 1.0107016e-07, "wooly"=> 1.4223620000000001e-07, "woons"=> 3.210398e-09, "woops"=> 9.905996e-09, "woopy"=> 5.471458e-10, "woose"=> 1.5520959999999998e-09, "woosh"=> 3.286762e-08, "wootz"=> 8.795906000000001e-09, "woozy"=> 3.988652e-07, "words"=> 0.0002633512, "wordy"=> 3.312536e-07, "works"=> 0.0001110396, "world"=> 0.000550316, "worms"=> 4.152878e-06, "wormy"=> 9.709243999999999e-08, "worry"=> 4.443928e-05, "worse"=> 5.138892000000001e-05, "worst"=> 3.12257e-05, "worth"=> 6.346285999999999e-05, "worts"=> 4.427548e-08, "would"=> 0.0017049320000000004, "wound"=> 2.73388e-05, "woven"=> 5.161908e-06, "wowed"=> 1.1157299999999999e-07, "wowee"=> 9.37641e-09, "woxen"=> 2.632008e-09, "wrack"=> 2.167842e-07, "wrang"=> 5.724734000000001e-08, "wraps"=> 2.64953e-06, "wrapt"=> 2.1952800000000003e-07, "wrast"=> 6.155737999999999e-10, "wrate"=> 3.1782159999999998e-09, "wrath"=> 8.685152e-06, "wrawl"=> 1.9633972e-10, "wreak"=> 7.418119999999999e-07, "wreck"=> 4.713624000000001e-06, "wrens"=> 2.007678e-07, "wrest"=> 6.534344e-07, "wrick"=> 2.8271570000000003e-09, "wried"=> 2.1030039999999995e-09, "wrier"=> 1.3691799999999998e-09, "wries"=> 7.938424000000001e-10, "wring"=> 8.154023999999999e-07, "wrist"=> 1.361658e-05, "write"=> 8.124126e-05, "writs"=> 5.180896e-07, "wroke"=> 6.50219e-10, "wrong"=> 0.000107048, "wroot"=> 5.813712e-09, "wrote"=> 7.987140000000001e-05, "wroth"=> 6.921316e-07, "wrung"=> 1.514482e-06, "wryer"=> 9.346616000000003e-10, "wryly"=> 1.310946e-06, "wuddy"=> 1.1798906e-09, "wudus"=> 5.974006e-11, "wulls"=> 2.825816e-10, "wurst"=> 1.0132312e-07, "wuses"=> 1.4808882000000002e-10, "wushu"=> 9.211544000000001e-08, "wussy"=> 2.4348759999999998e-08, "wuxia"=> 5.202906e-08, "wyled"=> 2.56122e-10, "wyles"=> 2.8450620000000002e-08, "wynds"=> 1.9895739999999998e-08, "wynns"=> 1.041998e-08, "wyted"=> 3.92947e-10, "wytes"=> 2.76067e-10, "xebec"=> 1.2480917999999998e-08, "xenia"=> 2.31404e-07, "xenic"=> 3.13795e-09, "xenon"=> 4.1924520000000006e-07, "xeric"=> 5.443622e-08, "xerox"=> 5.377942000000001e-07, "xerus"=> 7.089394e-09, "xoana"=> 5.172484e-09, "xrays"=> 5.930722000000001e-08, "xylan"=> 1.2177686000000001e-07, "xylem"=> 5.348912e-07, "xylic"=> 4.5112181999999993e-10, "xylol"=> 9.473776e-09, "xylyl"=> 6.5371899999999994e-09, "xysti"=> 1.0659362000000002e-09, "xysts"=> 0.0, "yaars"=> 7.360771999999999e-10, "yabas"=> 6.092830000000001e-10, "yabba"=> 1.137353e-08, "yabby"=> 1.1044877999999998e-08, "yacca"=> 1.9696779999999995e-09, "yacht"=> 3.82672e-06, "yacka"=> 8.323198e-10, "yacks"=> 9.568996e-10, "yaffs"=> 1.6074736e-09, "yager"=> 1.578382e-07, "yages"=> 2.293592e-10, "yagis"=> 2.1312e-09, "yahoo"=> 7.294296e-07, "yaird"=> 4.613884e-09, "yakka"=> 1.2920786e-08, "yakow"=> 2.469768e-10, "yales"=> 3.947804e-09, "yamen"=> 1.3447566e-07, "yampy"=> 2.5991000000000004e-10, "yamun"=> 2.9538139999999998e-09, "yangs"=> 1.700652e-08, "yanks"=> 7.646014e-07, "yapok"=> 1.996929e-09, "yapon"=> 4.3710259999999997e-10, "yapps"=> 2.5610242000000005e-10, "yappy"=> 4.694828e-08, "yarak"=> 7.162400000000001e-09, "yarco"=> 1.5710953999999999e-09, "yards"=> 1.69816e-05, "yarer"=> 1.1410618e-10, "yarfa"=> 4.1188139999999996e-10, "yarks"=> 3.7908359999999996e-10, "yarns"=> 1.093474e-06, "yarrs"=> 8.135794e-11, "yarta"=> 1.7890204000000002e-09, "yarto"=> 1.584828e-09, "yates"=> 1.964208e-06, "yauds"=> 4.86453e-10, "yauld"=> 8.338088000000002e-10, "yaups"=> 1.3937458e-10, "yawed"=> 4.5074480000000005e-08, "yawey"=> 3.3061294e-10, "yawls"=> 1.159188e-08, "yawns"=> 3.7357399999999993e-07, "yawny"=> 2.10995e-09, "yawps"=> 3.7465280000000005e-09, "ybore"=> 9.947688e-10, "yclad"=> 2.163336e-09, "ycled"=> 2.1435209999999999e-10, "ycond"=> 1.9948282000000002e-10, "ydrad"=> 3.80412e-10, "ydred"=> 9.739711999999999e-11, "yeads"=> 1.025816e-09, "yeahs"=> 2.8301239999999997e-08, "yealm"=> 3.1479239999999996e-09, "yeans"=> 7.727859999999999e-10, "yeard"=> 5.8391899999999996e-09, "yearn"=> 9.116104e-07, "years"=> 0.0004952026, "yeast"=> 5.806948e-06, "yecch"=> 2.1462820000000004e-09, "yechs"=> 1.0657579999999999e-10, "yechy"=> 1.2164702e-10, "yedes"=> 9.131308000000002e-10, "yeeds"=> 6.226356e-11, "yeesh"=> 2.18316e-08, "yeggs"=> 6.972014e-09, "yelks"=> 3.5495589999999996e-09, "yells"=> 2.266282e-06, "yelms"=> 1.7223724e-10, "yelps"=> 2.444816e-07, "yelts"=> 2.0112617999999997e-10, "yenta"=> 1.457938e-08, "yente"=> 6.740464e-09, "yerba"=> 1.565046e-07, "yerds"=> 5.893145999999999e-10, "yerks"=> 2.273154e-09, "yeses"=> 6.865028e-08, "yesks"=> 0.0, "yests"=> 9.854088000000001e-11, "yesty"=> 2.571838e-09, "yetis"=> 4.039223999999999e-08, "yetts"=> 4.730514e-09, "yeuks"=> 2.0135869999999996e-10, "yeuky"=> 1.2143834e-10, "yeven"=> 1.1321742000000001e-08, "yeves"=> 5.535542e-10, "yewen"=> 1.1306222e-09, "yexed"=> 6.718146e-11, "yexes"=> 1.1216522e-10, "yfere"=> 1.5996426e-09, "yield"=> 2.821524e-05, "yiked"=> 1.201376e-10, "yikes"=> 2.783786e-07, "yills"=> 1.7167172e-10, "yince"=> 5.3212060000000005e-09, "yipes"=> 7.746942000000001e-09, "yippy"=> 1.1775766e-08, "yirds"=> 1.2213486e-10, "yirks"=> 1.8309456e-10, "yirrs"=> 1.6330092000000003e-10, "yirth"=> 4.596446e-10, "yites"=> 2.6217996e-10, "yitie"=> 4.11734e-11, "ylems"=> 6.153466e-11, "ylike"=> 1.393376e-09, "ylkes"=> 5.0524760000000005e-11, "ymolt"=> 7.085688e-11, "ympes"=> 6.249746e-10, "yobbo"=> 6.786072e-09, "yobby"=> 7.722206e-10, "yocks"=> 7.077610000000001e-10, "yodel"=> 7.4139e-08, "yodhs"=> 2.3844464e-10, "yodle"=> 2.0457958e-09, "yogas"=> 7.159736e-08, "yogee"=> 4.8326914e-09, "yoghs"=> 3.563224e-10, "yogic"=> 4.028458e-07, "yogin"=> 7.55015e-08, "yogis"=> 3.23724e-07, "yoick"=> 1.3150862e-09, "yojan"=> 1.5729193999999998e-09, "yoked"=> 4.27825e-07, "yokel"=> 9.694046e-08, "yoker"=> 2.577276e-09, "yokes"=> 2.177016e-07, "yokul"=> 2.7148079999999997e-10, "yolks"=> 1.090095e-06, "yolky"=> 1.1877561999999998e-08, "yomim"=> 5.792674e-10, "yomps"=> 4.243954e-10, "yonic"=> 3.7835179999999995e-09, "yonis"=> 6.321398e-09, "yonks"=> 1.63656e-08, "yoofs"=> 1.6869416e-10, "yoops"=> 7.239138e-10, "yores"=> 5.980257999999999e-09, "yorks"=> 1.4632139999999997e-07, "yorps"=> 1.520229e-10, "youks"=> 1.0894018000000001e-09, "young"=> 0.0002534789999999999, "yourn"=> 1.1234037999999999e-07, "yours"=> 2.817482e-05, "yourt"=> 2.2763132e-09, "youse"=> 2.468508e-07, "youth"=> 5.3241140000000004e-05, "yowed"=> 1.7044740000000001e-10, "yowes"=> 5.816078000000001e-09, "yowie"=> 1.3247264000000001e-08, "yowls"=> 3.549462e-08, "yowza"=> 1.290636e-08, "yrapt"=> 1.2144542e-10, "yrent"=> 2.8999665999999994e-10, "yrivd"=> 0.0, "yrneh"=> 1.1907282e-10, "ysame"=> 2.4537664e-10, "ytost"=> 9.8823e-11, "yuans"=> 1.2400692e-08, "yucas"=> 1.7521584000000002e-09, "yucca"=> 3.5971480000000003e-07, "yucch"=> 4.1230179999999997e-10, "yucko"=> 1.1686962e-09, "yucks"=> 4.38125e-09, "yucky"=> 1.117622e-07, "yufts"=> 9.583886e-11, "yugas"=> 3.3113839999999997e-08, "yuked"=> 7.367854e-11, "yukes"=> 1.2916833999999999e-09, "yukky"=> 4.829114e-09, "yukos"=> 9.771906e-08, "yulan"=> 6.082833999999998e-08, "yules"=> 3.985129999999999e-09, "yummo"=> 1.454466e-09, "yummy"=> 6.578214000000001e-07, "yumps"=> 2.7681329999999996e-10, "yupon"=> 4.752517999999999e-10, "yuppy"=> 6.723542000000001e-09, "yurta"=> 4.3714204e-09, "yurts"=> 9.22423e-08, "yuzus"=> 2.6914591999999997e-10, "zabra"=> 7.1159874e-09, "zacks"=> 7.05176e-08, "zaida"=> 5.390832e-08, "zaidy"=> 2.4584250000000002e-09, "zaire"=> 6.61291e-07, "zakat"=> 3.254002e-07, "zaman"=> 5.303442e-07, "zambo"=> 4.2350040000000005e-08, "zamia"=> 2.3115376e-08, "zanja"=> 1.3913229999999997e-08, "zante"=> 5.3320079999999996e-08, "zanza"=> 1.45181e-09, "zanze"=> 2.9899520000000002e-09, "zappy"=> 4.85839e-09, "zarfs"=> 3.000728e-10, "zaris"=> 5.582702e-10, "zatis"=> 8.887268e-11, "zaxes"=> 9.783348000000002e-10, "zayin"=> 2.43261e-08, "zazen"=> 1.400924e-07, "zeals"=> 2.5367299999999998e-09, "zebec"=> 3.392996e-09, "zebra"=> 1.0607794e-06, "zebub"=> 4.882358e-08, "zebus"=> 7.675578e-09, "zedas"=> 4.216066e-11, "zeins"=> 6.601214e-09, "zendo"=> 3.0901240000000004e-08, "zerda"=> 9.164762e-09, "zerks"=> 1.3713785999999998e-09, "zeros"=> 1.7722259999999998e-06, "zests"=> 1.40792e-08, "zesty"=> 1.614088e-07, "zetas"=> 1.780722e-07, "zexes"=> 0.0, "zezes"=> 1.4598724000000001e-10, "zhomo"=> 7.128763999999999e-11, "zibet"=> 4.6409099999999995e-10, "ziffs"=> 4.149948e-10, "zigan"=> 2.4193460000000003e-09, "zilas"=> 4.7106262e-09, "zilch"=> 1.2919763999999998e-07, "zilla"=> 9.651178000000001e-08, "zills"=> 1.4441429999999999e-09, "zimbi"=> 1.7140016e-08, "zimbs"=> 3.425628e-11, "zinco"=> 5.4730099999999995e-09, "zincs"=> 4.076476e-09, "zincy"=> 2.4669144e-10, "zineb"=> 2.45787e-08, "zines"=> 1.39356e-07, "zings"=> 3.2648740000000007e-08, "zingy"=> 4.604948e-08, "zinke"=> 3.6900959999999995e-08, "zinky"=> 2.479038e-09, "zippo"=> 1.4561719999999998e-07, "zippy"=> 1.1664245999999998e-07, "ziram"=> 7.282476e-09, "zitis"=> 8.772596e-10, "zizel"=> 2.0859121999999998e-10, "zizit"=> 1.2251616e-09, "zlote"=> 4.174262e-10, "zloty"=> 5.653408e-08, "zoaea"=> 5.5274706e-10, "zobos"=> 1.7912508e-10, "zobus"=> 5.04304e-10, "zocco"=> 5.025204e-09, "zoeae"=> 5.364136e-09, "zoeal"=> 9.431145999999999e-09, "zoeas"=> 1.0156587999999998e-09, "zoism"=> 2.342872e-10, "zoist"=> 7.2364680000000005e-09, "zombi"=> 3.652448e-08, "zonae"=> 4.4446239999999995e-09, "zonal"=> 5.638338e-07, "zonda"=> 7.462427999999998e-09, "zoned"=> 4.397734000000001e-07, "zoner"=> 3.7039840000000003e-09, "zones"=> 1.234082e-05, "zonks"=> 9.721446e-10, "zooea"=> 1.3362368e-10, "zooey"=> 7.614746e-08, "zooid"=> 2.1034976000000002e-08, "zooks"=> 1.60644e-08, "zooms"=> 3.209372e-07, "zoons"=> 1.0849232e-08, "zooty"=> 1.0037616e-09, "zoppa"=> 1.96392e-09, "zoppo"=> 2.0802959999999998e-08, "zoril"=> 1.1351942000000002e-09, "zoris"=> 1.747714e-09, "zorro"=> 1.999778e-07, "zouks"=> 4.2048080000000003e-10, "zowee"=> 1.8375226e-09, "zowie"=> 2.3149119999999996e-08, "zulus"=> 2.8811440000000003e-07, "zupan"=> 4.865664e-08, "zupas"=> 1.9390562e-10, "zuppa"=> 3.1843940000000004e-08, "zurfs"=> 7.34093e-11, "zuzim"=> 9.151102e-09, "zygal"=> 1.1529654e-09, "zygon"=> 5.496122e-08, "zymes"=> 4.8407719999999994e-09, "zymic"=> 1.0911232000000001e-10)

# ╔═╡ 8c231aac-dfac-452b-aa17-240dfa2de317
get_word_freq(word) = lookup_dict(word_frequencies, word, 0.0)


# ╔═╡ c68def77-cb78-43ef-87ae-39a1c594b346
const freq_weights = [get_word_freq(w) for w in nyt_valid_words]


# ╔═╡ f290c009-924c-4d58-bd94-df89b118ec90
const freqsortinds = sortperm(freq_weights, rev = true)


# ╔═╡ 6e038c56-f55c-40b9-a456-4e15ad40d124
const word_freq_rank = Dict(zip(freqsortinds, eachindex(freqsortinds)))


# ╔═╡ d4837f88-5138-4e29-ae4c-436ee147ae5f
const wordranks = [word_freq_rank[word_index[w]] for w in wordle_original_answers] |> sort


# ╔═╡ ab9fe475-1651-4bb9-92a8-ee0c21c5e983
const excludewords = setdiff(setdiff(nyt_valid_words[freqsortinds][1:4000], wordle_original_answers), includewords)

# ╔═╡ 122d18fe-cc0b-4ccf-b0ae-fd78539bf611
const excludeinds = Set(word_index[w] for w in excludewords)

# ╔═╡ daafb692-2ef9-434a-8c9a-c1db735dc57b
make_pdf_weights(n) = map(eachindex(nyt_valid_words)) do i
	in(i, excludeinds) && return 0.0
	rank = word_freq_rank[i]
	rank > n && return 0.0
	wordle_pdf(rank)
end

# ╔═╡ 22bde1b7-493c-4bd3-9900-c3783bfc98c8
function normalize_vector!(v::AbstractVector) 
	s = sum(v)
	v ./= s
	return v
end

# ╔═╡ c96849c0-ce1f-4122-a8ce-f8337cc44640
const wordle_answer_weights = normalize_vector!(Float32.(make_pdf_weights(8000)))

# ╔═╡ bb1880c1-88a4-4f60-9003-d4b7473dbc09
#create a list of wordle answers for the game to use upon reset by sampling accoridng to the pdf from the valid guess list without replacement
make_pdf_answer_index(maxrank, numwords) = wsample(eachindex(nyt_valid_words), make_pdf_weights(maxrank), numwords, replace=false)

# ╔═╡ e82a3a4a-e248-4f23-a99a-261c795cfbc3
const wordle_original_pdf = make_pdf_weights(8000) |> normalize_vector!

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
AbstractPlutoDingetjes = "6e696c72-6542-2067-7265-42206c756150"
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
DataStructures = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
HypertextLiteral = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
JLD2 = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Markdown = "d6f4376e-aef5-505a-96c1-9c027394607a"
PlutoDevMacros = "a0499f29-c39b-4c5c-807c-88074221b949"
PlutoHooks = "0ff47ea0-7a50-410d-8455-4348d5de0774"
PlutoPlotly = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
PlutoProfile = "ee419aa8-929d-45cd-acf6-76bd043cd7ba"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"

[compat]
AbstractPlutoDingetjes = "~1.3.2"
BenchmarkTools = "~1.5.0"
DataFrames = "~1.6.1"
DataStructures = "~0.18.20"
HypertextLiteral = "~0.9.5"
JLD2 = "~0.4.52"
PlutoDevMacros = "~0.9.0"
PlutoHooks = "~0.0.5"
PlutoPlotly = "~0.5.0"
PlutoProfile = "~0.4.0"
PlutoUI = "~0.7.60"
StatsBase = "~0.34.3"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.10.5"
manifest_format = "2.0"
project_hash = "ceafd966df824b7570b8b1714b4bf92071f3ef4f"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.AbstractTrees]]
git-tree-sha1 = "03e0550477d86222521d254b741d470ba17ea0b5"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.3.4"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BaseDirs]]
git-tree-sha1 = "cb25e4b105cc927052c2314f8291854ea59bf70a"
uuid = "18cc8868-cbac-4acf-b575-c8ff214dc66f"
version = "1.2.4"

[[deps.BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "f1dff6729bc61f4d49e140da1af55dcd1ac97b2f"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.5.0"

[[deps.CodeTracking]]
deps = ["InteractiveUtils", "UUIDs"]
git-tree-sha1 = "7eee164f122511d3e4e1ebadb7956939ea7e1c77"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "1.3.6"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "b5278586822443594ff615963b0c09755771b3e0"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.26.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "b10d0b65641d57b8b4d5e234446582de5047050d"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.5"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "a1f44953f2382ebb937d60dafbe2deea4bd23249"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.10.0"

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

    [deps.ColorVectorSpace.weakdeps]
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "362a287c3aa50601b0bc359053d5c2468f0e7ce0"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.11"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "8ae8d32e09f0dcf42a36b90d4e17f5dd2e4c4215"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.16.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "DataStructures", "Future", "InlineStrings", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrecompileTools", "PrettyTables", "Printf", "REPL", "Random", "Reexport", "SentinelArrays", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "04c738083f29f86e62c8afc341f0967d8717bdb8"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.6.1"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "1d0a14036acb104d9e89698bd408f63ab58cdc82"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.20"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "82d8afa92ecf4b52d78d869f038ebfb881267322"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.16.3"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.FlameGraphs]]
deps = ["AbstractTrees", "Colors", "FileIO", "FixedPointNumbers", "IndirectArrays", "LeftChildRightSiblingTrees", "Profile"]
git-tree-sha1 = "d9eee53657f6a13ee51120337f98684c9c702264"
uuid = "08572546-2f56-4bcf-ba4e-bab62c3a3f89"
version = "0.2.10"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "b6d6bfdd7ce25b0f9b2f6b3dd56b2673a66c8770"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.5"

[[deps.IndirectArrays]]
git-tree-sha1 = "012e604e1c7458645cb8b436f8fba789a51b257f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "1.0.0"

[[deps.InlineStrings]]
git-tree-sha1 = "45521d31238e87ee9f9732561bfee12d4eebd52d"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.2"

    [deps.InlineStrings.extensions]
    ArrowTypesExt = "ArrowTypes"
    ParsersExt = "Parsers"

    [deps.InlineStrings.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"
    Parsers = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.InvertedIndices]]
git-tree-sha1 = "0dc7b50b8d436461be01300fd8cd45aa0274b038"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLD2]]
deps = ["FileIO", "MacroTools", "Mmap", "OrderedCollections", "PrecompileTools", "Requires", "TranscodingStreams"]
git-tree-sha1 = "049950edff105ff73918d29dbf109220ff364157"
uuid = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
version = "0.4.52"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JuliaInterpreter]]
deps = ["CodeTracking", "InteractiveUtils", "Random", "UUIDs"]
git-tree-sha1 = "4b415b6cccb9ab61fec78a621572c82ac7fa5776"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.9.35"

[[deps.LaTeXStrings]]
git-tree-sha1 = "50901ebc375ed41dbf8058da26f9de442febbbec"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.1"

[[deps.LeftChildRightSiblingTrees]]
deps = ["AbstractTrees"]
git-tree-sha1 = "b864cb409e8e445688bc478ef87c0afe4f6d1f8d"
uuid = "1d6d02ad-be62-4b6b-8a6d-2f90e265016e"
version = "0.1.3"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.4.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.6.4+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "a2d09619db4e765091ee5c6ffe8872849de0feea"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.28"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "2fa9ee3e63fd3a4f7a9a4f4744a52f4856de82df"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.13"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+1"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.1.10"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.23+4"

[[deps.OrderedCollections]]
git-tree-sha1 = "dfdf5519f235516220579f949664f1bf44e741c5"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.3"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.10.0"

[[deps.PlotlyBase]]
deps = ["ColorSchemes", "Dates", "DelimitedFiles", "DocStringExtensions", "JSON", "LaTeXStrings", "Logging", "Parameters", "Pkg", "REPL", "Requires", "Statistics", "UUIDs"]
git-tree-sha1 = "56baf69781fc5e61607c3e46227ab17f7040ffa2"
uuid = "a03496cd-edff-5a9b-9e67-9cda94a718b5"
version = "0.8.19"

[[deps.PlutoDevMacros]]
deps = ["JuliaInterpreter", "Logging", "MacroTools", "Pkg", "TOML"]
git-tree-sha1 = "72f65885168722413c7b9a9debc504c7e7df7709"
uuid = "a0499f29-c39b-4c5c-807c-88074221b949"
version = "0.9.0"

[[deps.PlutoHooks]]
deps = ["InteractiveUtils", "Markdown", "UUIDs"]
git-tree-sha1 = "072cdf20c9b0507fdd977d7d246d90030609674b"
uuid = "0ff47ea0-7a50-410d-8455-4348d5de0774"
version = "0.0.5"

[[deps.PlutoPlotly]]
deps = ["AbstractPlutoDingetjes", "Artifacts", "BaseDirs", "Colors", "Dates", "Downloads", "HypertextLiteral", "InteractiveUtils", "LaTeXStrings", "Markdown", "Pkg", "PlotlyBase", "Reexport", "TOML"]
git-tree-sha1 = "653b48f9c4170343c43c2ea0267e451b68d69051"
uuid = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
version = "0.5.0"

    [deps.PlutoPlotly.extensions]
    PlotlyKaleidoExt = "PlotlyKaleido"
    UnitfulExt = "Unitful"

    [deps.PlutoPlotly.weakdeps]
    PlotlyKaleido = "f2990250-8cf9-495f-b13a-cce12b45703c"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.PlutoProfile]]
deps = ["AbstractTrees", "FlameGraphs", "Profile", "ProfileCanvas"]
git-tree-sha1 = "154819e606ac4205dd1c7f247d7bda0bf4f215c4"
uuid = "ee419aa8-929d-45cd-acf6-76bd043cd7ba"
version = "0.4.0"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "eba4810d5e6a01f612b948c9fa94f905b49087b0"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.60"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "36d8b4b899628fb92c2749eb488d884a926614d3"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.3"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "9306f6085165d270f7e3db02af26a400d580f5c6"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.3"

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "66b20dd35966a748321d3b2537c4584cf40387c7"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "2.3.2"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Profile]]
deps = ["Printf"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

[[deps.ProfileCanvas]]
deps = ["FlameGraphs", "JSON", "Pkg", "Profile", "REPL"]
git-tree-sha1 = "41fd9086187b8643feda56b996eef7a3cc7f4699"
uuid = "efd6af41-a80b-495e-886c-e51b0c7d77a3"
version = "0.1.0"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "ff11acffdb082493657550959d4feb4b6149e73a"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.5"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "66e0a8e672a0bdfca2c3f5937efb8538b9ddc085"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.10.0"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.10.0"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "5cf7606d6cef84b543b483848d4ae08ad9832b21"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.3"

[[deps.StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "a04cabe79c5f01f4d723cc6704070ada0b9d46d5"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.3.4"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.2.1+1"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "598cd7c1f68d1e205689b1c2fe65a9f85846f297"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.12.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TranscodingStreams]]
git-tree-sha1 = "e84b3a11b9bece70d14cce63406bbc79ed3464d2"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.2"

[[deps.Tricks]]
git-tree-sha1 = "7822b97e99a1672bfb1b49b668a6d46d58d8cbcb"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.9"

[[deps.URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.11.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.52.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"
"""

# ╔═╡ Cell order:
# ╟─5d564193-c55b-4584-96ff-5b1cf404e334
# ╟─b04c7371-c26a-4400-8dae-06b922b27af1
# ╟─589fac35-d61b-4ece-a316-610b91f26640
# ╟─959f4088-9a24-4104-ad2e-1d1a8edfc3b2
# ╟─67d4a8f0-f96e-45f1-a8c0-25a31707551c
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
# ╟─e5b1d8e5-f224-44e3-8190-b8146ed3ea92
# ╟─32c30d8e-de5b-43a0-bfdb-8fb86037de7f
# ╠═1967f344-0d2e-4af0-a6b5-d8489079629d
# ╠═5003d34f-39b1-48f0-9ee9-5d38c1c2a5f0
# ╠═16754304-c592-44f3-baec-94a4d824eb49
# ╠═5eeed71e-4171-4063-89ee-90cfa5934413
# ╠═5d87942f-188a-437e-b923-7e91b9f5b923
# ╠═64dbd6c6-2691-4580-97e3-bb2f875472d7
# ╠═c1858879-2a20-4af6-af4c-a03d26dda7a3
# ╟─be061e94-d403-453a-99fb-7a1e13bebf52
# ╠═b75bb4fe-6b09-4d7c-8bc0-6bf224d2c95a
# ╠═783d068c-77b0-43e1-907b-e532317c5afd
# ╠═bb5ef5ff-93fe-4985-a920-442862e4498b
# ╟─b9a5d547-79b4-4c7c-a3eb-ed7e87513a88
# ╠═1d5ba870-3110-4576-a116-a8d0a4d84edc
# ╠═93b857e5-72db-4aaf-abb0-295beab4073c
# ╠═852dff48-8c1d-42f2-925e-53f889b1ac6a
# ╠═4eb18ec4-b327-4f97-a380-10469441cff8
# ╠═20a41989-6650-41dd-b34e-fa23c993e669
# ╠═d9c4ff04-12ba-4e1c-bc52-4c12388d514b
# ╠═97d40991-d719-4e87-bd70-94551a645448
# ╟─92cb4e96-714d-425d-a64b-eff26a5f92ef
# ╠═a383cd80-dd49-4f9f-ba67-1f58ea7eb0b6
# ╠═3937537e-1de7-4212-b939-0b37e315ffbf
# ╟─67b9207a-5004-4488-a8e5-43465adbc26b
# ╟─ee740b4f-93e0-4197-986f-d1ba47d23266
# ╟─2cda2a2c-caf2-474c-90a4-388f15260501
# ╟─f2ab3716-19a2-4cbb-b46f-a13c297e086d
# ╠═66105fd7-aa54-4007-a346-67f0ac7e1188
# ╠═b1e3f7ea-e2de-4467-b301-0c3cf225e433
# ╠═35f4bf5e-0da0-4d0f-a5d8-956d773e716e
# ╠═bd744b6d-db57-42bf-88c0-c667dfb03f5f
# ╟─8e2477a9-ca14-4ecf-9194-9bac5ef25fd4
# ╟─70dcac79-2839-4c1d-8f48-cbe294b3ffb0
# ╟─bf1c13cd-a066-4865-b43f-0cba3d3df8d2
# ╟─69aacb14-52b2-49a2-a2bc-e8c07b9e1986
# ╟─52faa94f-e7ca-4e28-9ed2-0a788db5e231
# ╠═3c2fb945-0173-4de5-9fdf-ea8b9da3cbd3
# ╠═d2bf55fa-ac04-45d6-9d6a-8c7ac855c3a1
# ╠═961593ef-964c-4363-b4ba-0d45cfe3d198
# ╠═c9b7a2bd-6f77-4df0-a50e-b71446b8a274
# ╠═d3d0df67-ec26-4352-9164-a53d14d1b065
# ╠═f6253866-7bef-400f-9713-e0fd5054b201
# ╠═b43a3b03-2cb4-4bd7-8476-d5b1480adc0b
# ╠═9a75b05d-82fd-4a3c-8cb9-5161dfc18949
# ╠═8ab11beb-930a-43cb-9a52-f19f49819f1b
# ╠═7a9e09f2-70ca-4180-9c1c-56d74e743098
# ╠═16f1c036-90c1-4784-b750-3293255e31f7
# ╠═b301b451-3276-45fa-8777-eb2069b3e580
# ╠═c58f7788-5d64-45af-b5a1-9b30c24f730c
# ╠═1eb7de02-ab3c-4c96-82f5-a09c163faa30
# ╠═2ae69afa-42e5-4c3e-ada0-a71cb288dfb0
# ╠═05ffe4f9-8cd6-424e-8955-e3ca21c486f3
# ╠═50b2a4d0-3f0e-49c6-bc8d-b490b2542bb0
# ╠═2b8af0f4-d71e-4ff6-aad8-fd54463d587c
# ╠═aaefe619-c7bb-4c1e-acfb-1b2f05ef388d
# ╠═8cb865b4-8928-4210-96d5-6d6a71deb03b
# ╠═09c89ae6-83e2-4f86-8c3f-b1528235c70a
# ╠═9df47c03-18c5-4e22-a6f9-a55ab5e35c39
# ╠═2fa84583-4ff7-48d0-bd62-cb7380535e91
# ╟─83302fc6-49ff-4d25-9fd7-0b491b79fc73
# ╟─01be89b2-6233-4e11-b142-fa6ad05880a3
# ╟─4578c627-127e-4a18-972c-03b38cff371b
# ╟─57281f5c-4c93-4308-b1e7-e5a890655262
# ╟─3b32e495-98ca-472d-a928-c76fb9302a57
# ╟─5a0ab378-2c88-4851-99a0-ab816457cc6b
# ╠═24478951-775b-4e05-88b5-dca8d70e1103
# ╟─9f9b8db2-7c74-49f0-b067-bb6aa433fbe0
# ╟─faade165-8066-4deb-93a4-2b7daabd57dc
# ╠═e804f5fc-c817-4851-a14b-fdcca1180773
# ╟─7f1f2057-24ca-44c0-8496-b6ef68d895bd
# ╠═4de12ef4-4e40-41da-aede-e72f8206f173
# ╠═3d935fe6-16d9-4fce-8a2d-33c763801b94
# ╠═e093dd57-040b-4983-a95f-057097fcefff
# ╠═e92e232e-9e8c-4b84-aa1b-49a67a079380
# ╠═6fdc99b1-beea-4c45-98bb-d257501a6878
# ╟─67d76524-e5b6-48bc-9add-2aefd333876d
# ╟─2bae314a-e97c-41ef-b1a4-187617d9c88b
# ╟─a6c2eb9a-32cb-4343-8e40-fa2b27d4d792
# ╟─57d62f1e-6213-4328-94ac-b4b3126ddd5b
# ╟─1c43214e-951d-4b66-8db5-4a90d40ab533
# ╟─b61dfb6f-b2d8-48e1-99b4-345a05274dc5
# ╟─4130d798-e202-4280-b876-9ca989a45a58
# ╟─2c1a68f8-19e7-4224-8c26-0f5704b07389
# ╟─e610c03c-4bd9-40e1-a019-87c4fce39602
# ╠═11fba9cf-1094-4deb-8e1a-ddd781314aa9
# ╠═a7bca65e-e932-4bee-aa4a-bd6da2215472
# ╠═a00b4f9b-eb74-4fa1-bc70-a97844b59022
# ╠═cdec4e37-6cfc-40c3-9fcc-e7bcb7cbe6e0
# ╠═6abd318a-9c0b-457e-aac2-c70a580c66cd
# ╠═e08de788-8947-4882-ae51-f7cbb0daa83b
# ╠═58ce6598-0cf0-4450-b50f-afc2da287755
# ╠═7de01236-ba21-4329-815e-beb4633882db
# ╠═c6fdb911-5042-4f88-8484-2dbcd553892e
# ╠═3685266e-6ba9-400b-a590-69f930650124
# ╠═4d3a8b0a-4116-4cc6-9ef4-bc1a405db8bb
# ╠═a9e89449-74ea-45d1-93eb-48d11903d03c
# ╠═06138a4e-67a3-47a3-84a5-92013fd404ca
# ╠═10d1f403-34c8-46ce-8cfc-d289608f465c
# ╟─caaeeaee-94b7-498d-a746-a2c5c7177347
# ╟─fb8439a3-ee95-487f-a755-ffcbd8b3c381
# ╟─d9d54e56-a32f-4ffb-820d-df0b7918c78c
# ╟─7cd9cb06-4731-4eaa-b745-32118278d360
# ╠═41e0cddb-e78d-477b-849a-124754340a3c
# ╠═03bbb910-dfa7-4c28-b811-afa9e5ca0e63
# ╠═cb3b46cf-8375-43b2-8736-97882e9b5e18
# ╠═1ba2fd82-f651-43b3-9411-753eef787b68
# ╠═0812f3c2-35ab-4e2d-87c0-35a7b44af6d4
# ╠═d2052e0c-c506-45b5-8deb-76d5e60d300e
# ╠═9ee34b0a-2e11-403f-839b-4e9991bc0eac
# ╠═7d4f39fc-0669-4ded-a890-780d3c6b8e70
# ╠═164fccfc-56c5-4538-acf1-90ec13db38f8
# ╠═b6b40be8-90e3-4335-93ef-f8d92ef1676d
# ╠═0da02107-ad39-4c43-9dfc-2e68736d8063
# ╠═430ee1a8-8267-4a72-8380-e7460a28e47e
# ╟─9b1c9eb1-f97d-44f5-9050-9b18ea8814b0
# ╟─a0ecb0ec-f442-45ca-bc15-9967e82a9905
# ╠═8a9ed8ec-10b6-43e8-bb9a-e8eba96f3ea0
# ╠═a6eb37ac-09fc-4f24-b987-0f1e200dfebc
# ╠═36fb4201-8261-4551-ae38-eba073e3046b
# ╠═b3a7619f-82ac-4a6b-9206-ed1b5cfa0078
# ╟─12126ca1-b728-4a91-bc53-f0dacd412265
# ╟─bfb77472-6bcc-4a9d-9cd1-a7c2686da539
# ╟─d2cf651d-3547-4b80-b837-3b5f297fffa5
# ╟─6d79c363-2ff5-4489-9870-3a3813f592f3
# ╠═a5befb7c-a1e1-4d55-9b8f-c599748f6f00
# ╟─1642481e-8da9-475c-98b4-92e36d90065b
# ╟─c8d9a991-ff7b-448e-820c-a1f4a89ee22e
# ╠═b9a41bbb-8706-4f1d-beb7-f55ad8ad5ce7
# ╠═71e35ad3-1c42-4ffd-946b-5eb9e6b72f86
# ╠═8327c794-200f-400f-8bdd-d043c548522c
# ╟─a107e7ff-56e8-4f31-b061-a7895ea29965
# ╟─bd37635d-5f92-4fe4-9245-44a4019fcd54
# ╟─9f132408-31f3-4ad3-a486-f6a94b276f32
# ╟─e1782372-cb2b-440c-80f1-27d42f31bf57
# ╠═23bb63d2-4287-40b8-af9b-89cb98185f17
# ╟─28b733b6-3989-493a-a00d-24c48da6b338
# ╟─0a4ac51e-ab8f-467f-9281-691023af400a
# ╟─b9e9f12d-aa40-49d2-9dc2-ac6110d869d7
# ╟─90c5ee33-b6ae-4cde-b554-cedba77c0e0c
# ╠═1e4b84d7-6fe9-4b1e-9f14-3a663421cb1f
# ╠═d4378e47-e231-495c-8e72-de864097421e
# ╠═6957a643-0240-425b-b167-ae290b696f16
# ╠═13044009-df94-4dd1-93b2-a774015ab1de
# ╠═052913b1-1f6f-49b6-bf4d-cefef430fea9
# ╟─9711392c-33e0-4281-a871-7216dc146de6
# ╠═b88f9cc7-6232-41b3-92b3-3e920d446a5f
# ╠═839c4998-ac37-4964-8590-11d4840c83aa
# ╠═6cb17eea-2ec9-48d5-9900-635687d5300f
# ╠═5d8c0ce9-275b-4a9e-afd3-4c6028b0600c
# ╠═4dd064bc-db44-4b18-90ed-5bc67c12774e
# ╠═3cb9ba3c-1b9c-4a6c-b8f2-cade651df78d
# ╠═6670e6d3-ff00-4a91-98c2-830c3aed92de
# ╠═6c514cb4-03d9-4108-aca3-489212bea90e
# ╠═d5a9ab96-beb1-4f72-b049-2795ee0c5940
# ╠═a41be793-431d-42f0-885b-7ff89efa0252
# ╠═460d6485-b13e-4deb-a0bd-53b9ebe0c47f
# ╠═5df48271-77d3-4ab5-aba2-bd3cc0b9f078
# ╟─d68d26f3-112d-4542-aab8-0477369f3a52
# ╟─f82f878f-0dfb-4d87-94ff-1b5564e30f5c
# ╟─50a41592-576b-424c-9513-ff1ea4c8ca0f
# ╟─5724fed9-8f1e-4fad-b631-88fe86354e14
# ╟─f2106242-c6f9-4bb2-85f0-059b3e9db621
# ╟─a0433105-3429-4b03-9f26-b01af2dc3979
# ╠═013d1153-d3c0-405c-8797-3419ffeb4d4f
# ╠═65334860-bd8b-4c08-a30a-c0baf80280ce
# ╠═0cff6b40-02fa-4916-b710-98ad589b799d
# ╠═c4946f7f-5d17-4a66-a493-b2a2698dc1d5
# ╠═ed8472be-d0e2-4e9d-b752-fd64250e1db2
# ╠═8895b0b9-9fd1-4f5d-8132-564d9fecbf3d
# ╠═50135114-e4f8-4042-805a-05425318f55e
# ╠═e1252197-4592-4485-8390-40253c01f6b6
# ╠═b0b4164d-7299-4e5d-a993-01ffafefa43b
# ╠═c52535b6-cfbe-4207-8d7d-1787cb58c3ca
# ╠═43ccabd2-3688-4016-88d7-29a26d46986c
# ╠═9a8b5ba7-45c0-4190-b047-5e9c9e10c530
# ╠═e7b49bac-0785-4311-a2e9-fa618bcf5eb7
# ╠═db05290f-ae09-4321-9979-dda7ddb3bb18
# ╠═70b62b2d-c833-4b58-9f86-81980d3dd06a
# ╠═f4631d73-9465-479d-828d-6924d03a6cda
# ╠═3f73964c-c171-4759-888e-af43e53b4b2e
# ╠═3424be60-3f82-450e-9293-7b1f58542f7e
# ╠═66baf9cc-a76c-429d-b143-f46e82134602
# ╠═3b765ff5-5d48-4525-9e3b-d374815eb4c4
# ╠═e3a0438a-24cf-498e-bb38-362e678e0992
# ╠═0cb3c137-99f4-4ec7-95c4-1ec3e1322d3d
# ╠═af66fab5-3921-4fdf-9820-ca6ad5a21769
# ╟─773f6fa9-8b07-40ad-9c01-cf1a27ceeca8
# ╠═2a125dec-d4bb-4214-bec8-db95821a6d45
# ╟─f1166557-4073-4d82-b4c0-db7189e7b381
# ╠═9fef413d-773a-432f-a3e0-780ec1431c1c
# ╠═1919ceb3-31aa-49f0-929d-3faa034e2f63
# ╠═b28792a8-319a-4b16-9f8d-eec2c4b4be49
# ╟─5c40b6c3-3638-4c26-9e74-235a6cb90078
# ╟─aec20c62-092a-4778-a9ed-4a7aacd24d50
# ╠═1593e79d-8571-4848-9ed4-9e44416dbd49
# ╠═a03295ca-646b-4548-ac62-c6e7a8b4b54d
# ╠═c715ce58-caf3-4fe3-b01f-60cc29dff1af
# ╠═0d712e31-4554-4f12-bd02-4386b5d06607
# ╟─bb131186-d25c-465a-9c55-46dc2a2f0256
# ╠═1817dd59-2cbd-4704-88dd-e3ec761c9163
# ╠═98ea8bac-11a6-4e52-a197-47b190a306a6
# ╟─3e2e41f9-95b5-4eaa-81a6-3a9669a55c1a
# ╟─789f5070-dd89-4884-9d7c-6c08843c5711
# ╟─955df8dd-a52b-4df9-a941-799a9e7d2d27
# ╠═a7dfa2f2-724a-40c0-94dc-0aeeba6c40ac
# ╠═510e77ba-ac46-475d-a7e1-a734e31f4bcc
# ╟─79a43e2d-e34d-4821-b8e3-bbe4aa33cca0
# ╟─c52a6849-8201-4153-99c0-c46c43622958
# ╟─17daab93-b3ad-4eea-a032-05e61b2fb690
# ╠═72ec3834-da5d-4986-84d2-fc5047719aa5
# ╠═7f60a260-b073-414a-8f4a-ead768ca2b16
# ╠═711bc605-7b7e-4fc8-9eec-c5afada0c0cd
# ╠═fa50ef5e-6f9f-42c6-9d04-fc7512c59bd9
# ╠═365f112b-8754-4cb3-8521-51d87f8820bd
# ╠═2d8ad341-f825-42a9-b796-07a0184f8a89
# ╠═b4b39f69-7c0d-438e-98eb-c07863c8b7e8
# ╠═122e878c-75ad-40ef-9061-004acb7a301a
# ╠═b7c1aee0-bff2-4440-a386-41f83809ee9f
# ╠═0a05e6da-1965-4803-b5f9-1816076b7ee3
# ╠═5503dee5-32c5-4c11-b761-921fab0fb1bb
# ╠═54e18f37-7dfc-4c81-9893-344331b32dd7
# ╠═9153e796-3185-48fc-bff6-f07078bdfc14
# ╠═eabcf79e-67d3-478f-abbc-cd8ba04138e5
# ╟─3153a447-9f56-42a2-aa19-95e8a230ac22
# ╠═9101a345-650e-42c6-93d0-515f92f5c72f
# ╠═eb2afee9-6c33-4e66-99eb-0899dedaa2a6
# ╠═ff70b9ad-18e5-4b9f-ade4-888de0fbbc5d
# ╠═66bb9ca5-93b1-4bec-8eb7-5581c44b56cd
# ╟─5cb34251-a56e-4002-8e25-35932996502a
# ╟─3e16c985-37e6-49e1-9561-21c79552425c
# ╠═526d423e-c65a-49fd-b15b-e4598254af93
# ╟─f22fb00d-37f0-4a83-bf60-048b936bf04d
# ╠═a79b430a-84d7-4253-bfc5-ffb200b98767
# ╠═6c532310-1f80-419a-a74b-7814bad6a354
# ╟─aaf516b5-f982-44c3-bcae-14d46ad72e82
# ╠═16f8f288-9371-44ae-914d-4ea13fec98f6
# ╠═2deb31ad-ab39-4f27-885e-79bfccce97e3
# ╠═e247734e-d1e4-43f2-a74e-0d0bd5971a4b
# ╠═afbace21-a366-4ef1-9b50-198381293d22
# ╠═1faaa157-3d5f-450d-99e7-dae5d70501f9
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
# ╟─10eb71ed-df57-49b2-ae98-336191346b29
# ╠═729ef5f5-1538-48dc-b028-2bb112921fb2
# ╠═8d328017-7f7a-45d0-90bc-77ddd036101f
# ╠═014696a0-0568-4944-ad2f-1811b726b2ee
# ╠═b9beca61-7efc-4186-81c2-1ceda103c801
# ╠═bab1c451-bf1b-495c-9448-0be284063733
# ╠═ccca515b-17ad-4469-ac57-61c2f3f86e62
# ╟─67e57ae3-9834-43e6-ad46-91a6c046b69b
# ╠═1baafa2c-b087-41be-9e5f-6adea8a985fc
# ╟─d7b0a4ba-ba18-41ad-ade3-dde119f08a13
# ╠═7e8be36e-5237-49f6-95ec-b9cb24b32c34
# ╟─c809ed82-919c-4a7e-9acb-664499859760
# ╠═c9ab058c-a3db-4fcd-8ff9-2029e79fbfc6
# ╠═8c231aac-dfac-452b-aa17-240dfa2de317
# ╠═c68def77-cb78-43ef-87ae-39a1c594b346
# ╠═f290c009-924c-4d58-bd94-df89b118ec90
# ╠═6e038c56-f55c-40b9-a456-4e15ad40d124
# ╠═d4837f88-5138-4e29-ae4c-436ee147ae5f
# ╠═ab9fe475-1651-4bb9-92a8-ee0c21c5e983
# ╠═122d18fe-cc0b-4ccf-b0ae-fd78539bf611
# ╠═daafb692-2ef9-434a-8c9a-c1db735dc57b
# ╠═c96849c0-ce1f-4122-a8ce-f8337cc44640
# ╠═ea259831-7c83-4780-8670-ef997b51fea0
# ╠═ada75047-17da-4857-9c18-b4d696a50681
# ╠═4b0fd69e-682b-47f5-852a-7dbe8bc1a1dd
# ╠═d025366f-eb5c-4bb0-b8d7-24132e2c3d27
# ╟─b36e6991-7292-4290-9e28-85430fc0e897
# ╠═22bde1b7-493c-4bd3-9900-c3783bfc98c8
# ╠═bb1880c1-88a4-4f60-9003-d4b7473dbc09
# ╠═e82a3a4a-e248-4f23-a99a-261c795cfbc3
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002

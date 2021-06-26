from model import Model
from room import Room

lsi_rooms = [           # Can store this in another file
    Room(3.124),        # Made up numbers that don't do anything yet.
    Room(354324.324),
    Room(34.324),
    Room(324.234),
    Room(34.324),
]


rooms = lsi_rooms
model = Model(lsi_rooms)
current_temps = [255.0, 244.0, 234.0, 323.0, 234.1]

assert(len(current_temps) == len(rooms))

dt = 1  # Observable step in seconds (model may step more finely internally)
xs = current_temps
time = 0
while True:
    time += dt
    xs = model.evolve(xs, dt)
    print(time, " : ", xs)

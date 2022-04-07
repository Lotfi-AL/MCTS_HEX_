from tensorflow import keras
from config import HexConfig as config
from Actor import Actor
from timeit import default_timer as timer

anet = keras.models.load_model(config.LOAD_PATH + str(400))

actor = Actor(anet, greedy=True)

from ActorClient import ActorClient


class MyClient(ActorClient):
    def handle_get_action(self, state):
        start = timer()
        row, col = actor.choose_action(state)
        end = timer()
        print("took: ", end - start, "seconds")
        return row, col


if __name__ == "__main__":
    client = MyClient(auth="e9fd2e0c5dce408280b9913fca380424")
    client.run()

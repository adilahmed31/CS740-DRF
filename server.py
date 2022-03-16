import grpc
from concurrent import futures
import logging

import chord_pb2
import chord_pb2_grpc
import hashlib
import config
from chord import Node, Ring

def run_server(ip, id):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    node_dict = {}
    ring = Ring(node_dict)
    node = Node(ip, id, ring)
    chord_pb2_grpc.add_RPCServicer_to_server(node, server)
    server.add_insecure_port(ip)
    server.start()
    node.run()
    print("Running server on " + ip)
    server.wait_for_termination()

if __name__ == "__main__":
    run_server("127.0.0.1:50051",0)
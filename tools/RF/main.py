
import argparse
from concurrent import futures as fut
from typing import List, Tuple

import grpc
import joblib
import numpy as np
from shapely.geometry import LineString

import competition_pb2 as pb
import competition_pb2_grpc as pb_grpc
import sys
import competition_pb2

import pandas as pd
import shapely
from dataclasses import dataclass

from collections.abc import Mapping
# ========= Features possible=========




@dataclass
class TestDetails:
    test_id: str
    hasFailed: bool
    sim_time: float
    road_points: list[tuple[float, float]]


def _curvature_profile(test_detail: TestDetails) -> list[float]:
    """
    Compute the curvature for every meter of the road.

    The following website was used as a reference: https://de.wikipedia.org/wiki/Kr%C3%BCmmung
    """
    #print("compute curvature profile")
    road_shape = shapely.LineString(test_detail.road_points)

    delta_s = 2  # 10 meters

    curvature_profile = np.zeros(int(road_shape.length)) # we want the curvature for every meter
    for s in range(len(curvature_profile)):
        #s = (i+1)*delta_s

        # ignore the edge cases close to the ends of the road
        if (s < delta_s/2) or (s > road_shape.length-delta_s/2):
            continue


        pt_q: shapely.Point = road_shape.interpolate(s-delta_s, normalized=False)
        pt_r: shapely.Point = road_shape.interpolate(s-delta_s/2, normalized=False)

        pt_s: shapely.Point = road_shape.interpolate(s, normalized=False)

        pt_t: shapely.Point = road_shape.interpolate(s+delta_s/2, normalized=False)
        pt_u: shapely.Point = road_shape.interpolate(s+delta_s, normalized=False)

        tangent_r_vec = np.array((pt_s.x-pt_q.x, pt_s.y-pt_q.y))
        tangent_t_vec = np.array((pt_u.x-pt_s.x, pt_u.y-pt_s.y))

        cos_phi = np.dot(tangent_r_vec, tangent_t_vec)/(np.linalg.norm(tangent_r_vec)*np.linalg.norm(tangent_t_vec))
        phi = np.arccos(cos_phi)

        kappa = phi/delta_s
        if np.isnan(kappa):
            continue

        curvature_profile[s] = kappa

    return curvature_profile


def Radius_metric(radius: np.ndarray) -> dict:
    return radius[np.isfinite(radius)]


def _distance_profile(test_detail: TestDetails) -> list[float]:
    """
    Compute the cumulative distance along the road at each meter.

    This function returns the distance from the start of the road for each sampled point.
    """
    road_shape = shapely.LineString(test_detail.road_points)

    delta_s = 1  # Resolution to compute I put 1 meter to be more flexible 
    total_length = int(road_shape.length)

    distance_profile = np.zeros(total_length)

    for s in range(total_length):
        if s < 0 or s > road_shape.length:
            continue

        pt: shapely.Point = road_shape.interpolate(s, normalized=False)
        distance_profile[s] = s 

    return distance_profile



def build_feature_df(tests: List[TestDetails]):
    rows = []
    seuil = 1e-9
    for t in tests:
        kappa = np.asarray(_curvature_profile(t), dtype=float)
        radius = Radius_metric(kappa)
        dist = np.asarray(_distance_profile(t), dtype=float)
        rows.append({
            "test_id": t.test_id,
            "hasFailed": int(bool(t.hasFailed)),
            "sim_time": float(t.sim_time) if t.sim_time == t.sim_time else np.nan,
            ("curvature_mean", "curvature_std"): (np.mean(kappa), np.std(kappa)) , 
            ("dist_mean", "dist_std"): (np.mean(dist), np.mean(dist)) , 
            ("Radius_mean", "Radius_std"): (np.std(radius), np.std(radius))
        })
    return pd.DataFrame(rows)




class RFSelector(pb_grpc.CompetitionToolServicer):
    """
    Random Forest with only one metric : the curvature 
    """
    def __init__(self, model_path: str, service_name: str = "RF-Selector", k: int | None = None):
        bundle = joblib.load(model_path)
        print(bundle)


        self.rf = bundle["model"] if isinstance(bundle, Mapping) and "model" in bundle else bundle

        self.service_name = service_name
        self.k = k

        n_in = getattr(self.rf, "n_features_in_", None)
        self.n = n_in

        if n_in is not None and n_in not in (1, 2, 3):
            raise ValueError(
                f"Model expects {n_in} features; RFSelector supports only 1, 2, or 3."
            )

        if not hasattr(self.rf, "predict_proba"):
            raise TypeError("Loaded model has no predict_proba(). A classifier is required.")

    def Name(self, request: pb.Empty, context):
        return pb.NameReply(name=self.service_name)

    def Initialize(self, request_iterator, context):
        for _oracle in request_iterator:
            pass
        return pb.InitializationReply(ok=True)

    def Select(self, request_iterator, context):
        ids = []
        rows = []

        try:
            # 1) Ingest all test cases first (client will close the stream when done)
            for tc in request_iterator:
                # points from proto (support both camel/snake)
                pts = [(p.x, p.y) for p in getattr(tc, "roadPoints", [])] or \
                    [(p.x, p.y) for p in getattr(tc, "road_points", [])]

                # Build the TestDetails object used by your feature funcs
                td = TestDetails(
                    test_id=tc.testId,
                    hasFailed=False,
                    sim_time=float("nan"),
                    road_points=pts,
                )

                # 2) Build features (use means, adjust order to match training)
                kappa = np.asarray(_curvature_profile(td), dtype=float)

                with np.errstate(divide="ignore", invalid="ignore"):
                    radius_vals = np.where(kappa > 0, 1.0 / kappa, np.nan)
                radius = radius_vals[np.isfinite(radius_vals)]

                dist = np.asarray(_distance_profile(td), dtype=float)

                if self.n == 1:
                    fv = [float(np.mean(dist))]  # curvature_mean
                elif self.n == 2:
                    fv = [float(np.mean(dist)), float(np.mean(kappa))]  # dist_mean, curvature_mean
                else:
                    fv = [float(np.mean(dist)), float(np.mean(kappa)), float(np.mean(radius))]  # dist_mean, curvature_mean, radius mean

                ids.append(td.test_id)
                rows.append(fv)

        except Exception as e:
            context.set_details(f"Select() failed while building features: {e}")
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            return

        if not ids:
            return

        X = np.asarray(rows, dtype=float)
        if self.n is not None and X.shape[1] != self.n:
            context.set_details(f"Feature shape {X.shape} != expected (_, {self.n})")
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            return

        try:
            proba = self.rf.predict_proba(X)
            scores = proba[:, 1] if proba.shape[1] > 1 else np.zeros(len(ids), dtype=float)
        except Exception as e:
            context.set_details(f"Model prediction failed: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            return

        order = np.argsort(-scores)
        if self.k is not None:
            order = order[:self.k]

        # 4) Return only the chosen test cases (their IDs)
        for i in order:
            yield pb.SelectionReply(testId=ids[i])




# ========= SERVER MAIN =========
if __name__ == "__main__":
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", type=int, default=50051,
                        help="Port gRPC of the server")
    parser.add_argument("-m", "--model", default="rf_model.joblib",
                        help="model .joblib (default: rf_model.joblib)")
    parser.add_argument("-name","--name", default="test",
                        help="name of the test")
    parser.add_argument("--k", type=int, default=92,
                        help="max testing  to permit comparision between NN")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[ERREUR] no model: {model_path.resolve()}")
        sys.exit(1)

    GRPC_URL = f"[::]:{args.port}"  
    print("start test selector")
    print(f"  model: {model_path}")
    print(f"  top-k: {args.k if args.k is not None else 'no limit'}")

    server = grpc.server(fut.ThreadPoolExecutor(max_workers=2))


    servicer =  RFSelector(
        model_path=str(model_path),
        service_name = args.name,
        k=args.k
    )
    pb_grpc.add_CompetitionToolServicer_to_server(servicer, server)

    # Server Launching
    server.add_insecure_port(GRPC_URL)
    print(f"start server on {GRPC_URL}")
    server.start()
    print("server is running")
    server.wait_for_termination()

import argparse
from eval.surf_acc import MeshEvaluator


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_path', type=str, required=True, help='Directory to ground truth meshes')
    parser.add_argument('--pred_path', type=str, required=True, help='Directory to predicted meshes')
    parser.add_argument('--sample_num', type=int, default=100000, help='Number of points to sample [default: 100k]')
    args = parser.parse_args()

    evaluator = MeshEvaluator(sample_num=args.sample_num)
    
    # Use dense point clouds with oriented normals as the GT surface for CARLA
    if 'CARLA' in args.gt_path:
        evaluator.set_gt_surface_type('pointcloud')

    evaluator.run_batch_eval(args.gt_path, args.pred_path)

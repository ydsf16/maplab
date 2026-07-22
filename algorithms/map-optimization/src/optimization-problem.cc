#include "map-optimization/optimization-problem.h"

#include <memory>
#include <sstream>
#include <unordered_set>
#include <utility>
#include <vector>

#include <ceres-error-terms/parameterization/pose-param-jpl.h>
#include <ceres-error-terms/problem-information.h>
#include <vi-map-helpers/mission-clustering-coobservation.h>
#include <vi-map/vi-map.h>

#include "map-optimization/mission-cluster-gauge-fixes.h"
#include "map-optimization/optimization-state-buffer.h"

namespace map_optimization {
namespace {
void fixOpenDoFOfInitialVertex(
    double* first_pose_q_IM__M_p_MI_JPL_state,
    double* baseframe_q_GM__G_p_GM_JPL_state,
    FixedRotationDoF rotation_dof_fixed, bool position_dof_fixed,
    ceres_error_terms::ProblemInformation* problem_info) {
  CHECK_NOTNULL(first_pose_q_IM__M_p_MI_JPL_state);
  CHECK_NOTNULL(baseframe_q_GM__G_p_GM_JPL_state);
  CHECK_NOTNULL(problem_info);

  // Assemble a parameterization for the first vertex.
  ceres::LocalParameterization* position_parametrization = nullptr;
  if (position_dof_fixed) {
    position_parametrization =
        new ceres::SubsetParameterization(3, std::vector<int>{0, 1, 2});
  } else {
    position_parametrization = new ceres::IdentityParameterization(3);
  }

  ceres::LocalParameterization* rotation_parametrization = nullptr;
  switch (rotation_dof_fixed) {
    case FixedRotationDoF::kAll: {
      rotation_parametrization =
          new ceres_error_terms::JplQuaternionParameterization;
      problem_info->setParameterBlockConstant(
          first_pose_q_IM__M_p_MI_JPL_state);
      break;
    }
    case FixedRotationDoF::kYaw: {
      Eigen::Map<Eigen::Matrix<double, 7, 1>> q_GM__G_p_GM_JPL(
          baseframe_q_GM__G_p_GM_JPL_state);
      rotation_parametrization =
          new ceres_error_terms::JplRollPitchQuaternionParameterization(
              q_GM__G_p_GM_JPL.head<4>());
      problem_info->setParameterBlockVariable(
          first_pose_q_IM__M_p_MI_JPL_state);

      // This parameterization requires the baseframe to be constant.
      problem_info->setParameterBlockConstantIfPartOfTheProblem(
          baseframe_q_GM__G_p_GM_JPL_state);
      break;
    }
    case FixedRotationDoF::kNone: {
      rotation_parametrization =
          new ceres_error_terms::JplQuaternionParameterization;
      problem_info->setParameterBlockVariable(
          first_pose_q_IM__M_p_MI_JPL_state);
      break;
    }
    default:
      LOG(FATAL);
  }
  std::shared_ptr<ceres::LocalParameterization> first_pose_param(
      new ceres::ProductParameterization(
          rotation_parametrization, position_parametrization));
  problem_info->replaceParameterization(
      first_pose_q_IM__M_p_MI_JPL_state, first_pose_param);
}
}  // namespace

OptimizationProblem::OptimizationProblem(
    vi_map::VIMap* map, const vi_map::MissionIdSet& mission_ids)
    : map_(CHECK_NOTNULL(map)),
      missions_ids_(mission_ids),
      mission_coobservation_clusters_(
          vi_map_helpers::clusterMissionByLandmarkCoobservations(
              *map, mission_ids)) {
  state_buffer_.importStatesOfMissions(*map, mission_ids);

  // Initialize the parameterizations.
  local_parameterizations_.pose_parameterization.reset(
      new ceres_error_terms::JplPoseParameterization);
  local_parameterizations_.baseframe_parameterization.reset(
      new ceres_error_terms::JplYawOnlyPoseParameterization);
  local_parameterizations_.quaternion_parameterization.reset(
      new ceres_error_terms::JplQuaternionParameterization);
}

void OptimizationProblem::applyGaugeFixesForInitialVertices(
    const std::vector<MissionClusterGaugeFixes>& new_cluster_fixes) {
  CHECK_EQ(new_cluster_fixes.size(), mission_coobservation_clusters_.size());

  std::stringstream message;
  message << "VI gauge fixes: \n"
          << "  num mission co-observability clusters: "
          << new_cluster_fixes.size() << "\n";

  for (size_t cluster_idx = 0u; cluster_idx < new_cluster_fixes.size();
       ++cluster_idx) {
    const MissionClusterGaugeFixes& new_cluster_fix =
        new_cluster_fixes[cluster_idx];
    const vi_map::MissionIdSet& missionids_of_cluster =
        mission_coobservation_clusters_[cluster_idx];

    // Get the first vertex of the first mission of the cluster (which is also
    // part of the problem) and fix the open degrees of freedom.
    const vi_map::MissionId& first_mission_id = *missionids_of_cluster.begin();

    pose_graph::VertexId first_vertex_id_in_problem;
    double* first_vertex_state = nullptr;
    for (const vi_map::MissionId& mission_id : missionids_of_cluster) {
      pose_graph::VertexId current_vertex_id =
          map_->getMission(mission_id).getRootVertexId();
      do {
        CHECK(current_vertex_id.isValid());
        if (problem_books_.keyframes_in_problem.count(current_vertex_id) >
            0u) {
          double* candidate_state =
              state_buffer_.get_vertex_q_IM__M_p_MI_JPL(current_vertex_id);
          if (problem_information_.parameterizations.count(candidate_state) >
              0u) {
            first_vertex_id_in_problem = current_vertex_id;
            first_vertex_state = candidate_state;
            break;
          }
        }
      } while (map_->getNextVertex(current_vertex_id, &current_vertex_id));
      if (first_vertex_id_in_problem.isValid()) {
        break;
      }
    }
    CHECK(first_vertex_id_in_problem.isValid());
    CHECK_NOTNULL(first_vertex_state);
    const vi_map::MissionBaseFrameId baseframe_id =
        map_->getMissionBaseFrameForMission(first_mission_id).id();
    double* baseframe_state =
        state_buffer_.get_baseframe_q_GM__G_p_GM_JPL(baseframe_id);
    fixOpenDoFOfInitialVertex(
        first_vertex_state, baseframe_state,
        new_cluster_fix.rotation_dof_fixed, new_cluster_fix.position_dof_fixed,
        &problem_information_);

    message << "  cluster " << cluster_idx << ":\n"
            << "    missions of cluster: "
            << printIdContainer(missionids_of_cluster) << "\n"
            << "    1st mission: " << first_mission_id << "\n"
            << "    1st vertex: " << first_vertex_id_in_problem << "\n"
            << "    position-fixed: " << new_cluster_fix.position_dof_fixed
            << "\n"
            << "    rotation-fixed: " << new_cluster_fix.rotation_dof_fixed
            << "\n"
            << "    scale-fixed: " << new_cluster_fix.scale_fixed << "\n";

    // Fix scale of mission cluster by fixing a landmark expressed in the
    // vertex of the first vertex of the first mission in the cluster.
    if (new_cluster_fix.scale_fixed) {
      vi_map::LandmarkId first_landmark_of_first_mission;
      for (const vi_map::MissionId& mission_id : missionids_of_cluster) {
        pose_graph::VertexIdList vertex_ids;
        map_->getAllVertexIdsInMissionAlongGraph(mission_id, &vertex_ids);
        for (const pose_graph::VertexId& vertex_id : vertex_ids) {
          const vi_map::LandmarkStore& landmark_store =
              map_->getVertex(vertex_id).getLandmarks();
          for (const vi_map::Landmark& landmark : landmark_store) {
            if (problem_books_.landmarks_in_problem.count(landmark.id()) >
                0u) {
              first_landmark_of_first_mission = landmark.id();
              break;
            }
          }
          if (first_landmark_of_first_mission.isValid()) {
            break;
          }
        }
        if (first_landmark_of_first_mission.isValid()) {
          break;
        }
      }
      CHECK(first_landmark_of_first_mission.isValid())
          << "The mission cluster has no landmarks in the problem.";

      problem_information_.setParameterBlockConstantIfPartOfTheProblem(
          map_->getLandmark(first_landmark_of_first_mission).get_p_B_Mutable());
    }
  }
  if (problem_books_.cluster_gauge_fixes_initial_vertex == nullptr) {
    problem_books_.cluster_gauge_fixes_initial_vertex.reset(
        new std::vector<MissionClusterGaugeFixes>(
            mission_coobservation_clusters_.size()));
  }
  VLOG(2) << message.str();

  *problem_books_.cluster_gauge_fixes_initial_vertex = new_cluster_fixes;
}

}  // namespace map_optimization

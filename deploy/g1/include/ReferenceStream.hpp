#pragma once

#include <algorithm>
#include <deque>
#include <limits>
#include <mutex>
#include <set>
#include <nlohmann/json.hpp>
#include "G1Constants.hpp"

namespace teleop {
// All local times are steady-clock seconds. Remote timestamps are used only
// for differences: synchronization of workstation and robot clocks is unnecessary.
class ReferenceStream {
    struct Frame {
        Joints q;
        Mask active;
        Eigen::Quaterniond root;
        Eigen::Vector3d position{0.,0.,.7842};
        bool base_tracked=false;
        Eigen::Vector3d linear_velocity_world=Eigen::Vector3d::Zero();
        bool motion_reference=false;
        TorsoReference torso;
        double time, source_time;
        int64_t index;
    };
    mutable std::mutex mutex_;
    std::deque<Frame> frames_;
    std::deque<double> intervals_;
    Joints held_ = standing();
    Eigen::Quaterniond held_root_ = Eigen::Quaterniond::Identity();
    double last_received_ = -1., origin_local_ = 0., origin_source_ = 0.;
    double last_frame_received_ = -1.;
    double cursor_ = -std::numeric_limits<double>::infinity();
    double delay_floor_, timeout_, period_;
    std::string session_;
    nlohmann::json person_;
    bool stopped_ = false;
    uint64_t generation_ = 0;
    int64_t last_index_ = -1;
    double last_source_time_ = -std::numeric_limits<double>::infinity();
    double last_packet_ = -1., last_latency_ = -1.;
    std::string last_retarget_status_ = "none", last_tracking_reason_ = "not_reported";
    uint64_t packets_ = 0, valid_poses_ = 0, rejected_packets_ = 0, reference_holds_ = 0;
    int64_t last_valid_frame_ = -1;
    std::string last_status_ = "none", last_rejection_;

    void clear_frames() {
        frames_.clear(); intervals_.clear();
        cursor_ = -std::numeric_limits<double>::infinity();
        ++generation_;
    }
    double delay_unlocked() const {
        double interval = period_;
        for (double value : intervals_) interval = std::max(interval, value);
        return std::max(delay_floor_, interval + period_);
    }
    Reference at(double time) const {
        Reference out;
        out.q = held_; out.root = held_root_;
        if (frames_.size() < 2) return out;
        auto b = std::upper_bound(frames_.begin(), frames_.end(), time,
            [](double t, const Frame& f) { return t < f.time; });
        if (b == frames_.begin()) ++b;
        if (b == frames_.end()) b = frames_.end() - 1;
        const Frame& right = *b;
        const Frame& left = *(b - 1);
        out.alpha = std::clamp((time - left.time) / (right.time - left.time), 0., 1.);
        out.previous_frame = left.index; out.next_frame = right.index;
        out.source_time = left.source_time + out.alpha * (right.source_time - left.source_time);
        out.ready = true;
        for (int i = 0; i < 29; ++i) {
            // Disable immediately on the newest observation. Re-enable only
            // across two observed endpoints; never bridge an occlusion.
            out.active[i] = left.active[i] && right.active[i] && frames_.back().active[i];
            if (out.active[i]) {
                out.q[i] = left.q[i] + out.alpha * (right.q[i] - left.q[i]);
                if (time >= left.time && time < right.time)
                    out.dq[i] = (right.q[i] - left.q[i]) / (right.time - left.time);
            }
        }
        out.root = left.root.slerp(out.alpha, right.root).normalized();
        out.base_tracked=left.base_tracked&&right.base_tracked&&frames_.back().base_tracked;
        out.position=(1.-out.alpha)*left.position+out.alpha*right.position;
        out.motion_reference=out.base_tracked&&left.motion_reference&&right.motion_reference&&frames_.back().motion_reference;
        const double span=right.time-left.time;
        if(time>=left.time&&time<right.time){
            out.linear_velocity_world=(right.position-left.position)/span;
            auto delta=right.root*left.root.conjugate();if(delta.w()<0.)delta.coeffs()*=-1.;
            const Eigen::AngleAxisd aa(delta);out.root_angular_world=aa.axis()*aa.angle()/span;
        }
        if(out.motion_reference)out.linear_velocity_world=(1.-out.alpha)*left.linear_velocity_world+out.alpha*right.linear_velocity_world;
        out.torso.valid=out.motion_reference&&left.torso.valid&&right.torso.valid;
        if(out.torso.valid){
            out.torso.position=(1.-out.alpha)*left.torso.position+out.alpha*right.torso.position;
            out.torso.orientation=left.torso.orientation.slerp(out.alpha,right.torso.orientation).normalized();
            out.torso.linear=(1.-out.alpha)*left.torso.linear+out.alpha*right.torso.linear;
            out.torso.angular=(1.-out.alpha)*left.torso.angular+out.alpha*right.torso.angular;
        }
        return out;
    }
    Reference sample_unlocked(double now) {
        const double delay = delay_unlocked();
        if (!frames_.empty()) cursor_ = std::min(frames_.back().time, std::max(cursor_, now - delay));
        auto out = at(cursor_);
        out.delay = delay;
        out.stale = last_received_ < 0. || now - last_received_ > timeout_;
        out.stopped = stopped_;
        if (out.stale || out.stopped) { out.ready = false; out.active.fill(false); out.dq.setZero(); out.motion_reference=false; out.linear_velocity_world.setZero(); out.torso.valid=false; }
        if (out.ready) { held_ = out.q; held_root_ = out.root; }
        return out;
    }

public:
    struct Window { std::array<Reference, 10> frames; uint64_t generation; };
    ReferenceStream(double hz = 50., double delay_floor = 0., double timeout = 1.)
        : delay_floor_(delay_floor), timeout_(timeout), period_(1. / hz) {
        if (!std::isfinite(hz) || hz < 1 || hz > 1000 || !std::isfinite(delay_floor)
            || delay_floor < 0 || !std::isfinite(timeout) || timeout <= delay_floor + 1. / hz)
            throw std::invalid_argument("Invalid interpolation frequency/delay/timeout");
    }
    void reset(const Joints& measured) {
        std::lock_guard<std::mutex> lock(mutex_);
        clear_frames(); held_ = measured; held_root_ = Eigen::Quaterniond::Identity();
        last_received_ = last_frame_received_ = -1.; session_.clear(); person_ = nullptr; stopped_ = false;
        last_index_ = -1; last_source_time_ = -std::numeric_limits<double>::infinity();
        last_packet_ = last_latency_ = -1.; packets_ = valid_poses_ = rejected_packets_ = reference_holds_ = 0;
        last_valid_frame_ = -1; last_status_ = "none"; last_rejection_.clear();
        last_retarget_status_ = "none"; last_tracking_reason_ = "not_reported";
    }
    bool ingest(const nlohmann::json& packet, double received) {
        std::lock_guard<std::mutex> lock(mutex_);
        ++packets_;
        if (std::isfinite(received)) last_packet_ = received;
        auto reject = [this](const std::string& reason) {
            ++rejected_packets_; last_rejection_ = reason; return false;
        };
        try {
            if (!packet.is_object() || packet.value("schema", "") != "gmr-g1-v1"
                || packet.value("robot", "") != "unitree_g1" || !std::isfinite(received)) return reject("wrong_schema_or_robot");
            const auto status = packet.value("status", "");
            const bool fallback = status == "fallback";
            last_status_ = status;
            last_retarget_status_ = packet.value("retarget_status", status);
            last_tracking_reason_ = packet.value("retarget_reason", packet.value("tracking_reason", std::string("not_reported")));
            const std::string incoming_session = packet.value("stream_id", "legacy");
            if (!session_.empty() && incoming_session != session_ && status != "tracking" && status != "partial_tracking" && !fallback) return reject("control_packet_from_another_session");
            if (status == "stopped") { stopped_ = true; clear_frames(); return true; }
            if (status == "no_pose" || status == "invalid_pose") {
                if (packet.contains("frame")) {
                    if (!packet["frame"].is_number_integer()) return reject("invalid_frame_index");
                    const auto index = packet["frame"].get<int64_t>();
                    if (index < 0 || (incoming_session == session_ && index <= last_index_)) return reject("duplicate_or_out_of_order_frame");
                    if (incoming_session == session_) last_index_ = index;
                }
                clear_frames(); return true;
            }
            if (status != "tracking" && status != "partial_tracking" && !fallback) return reject("unknown_status");
            if (packet.value("joint_units", "") != "radians") return reject("joint_units_are_not_radians");
            const double latency = packet.value("latency_s", 0.);
            if (std::isfinite(latency)) last_latency_ = latency;
            if (!std::isfinite(latency) || latency < 0. || latency > timeout_) return reject("pose_latency_exceeds_timeout");
            auto names = packet.at("joint_names").get<std::vector<std::string>>();
            auto values = packet.at("joint_positions");
            auto active = packet.at("active_joint_mask").get<std::vector<bool>>();
            if (names.size() != 29 || values.size() != 29 || !values.is_array() || active.size() != 29
                || std::set<std::string>(names.begin(), names.end()).size() != 29) return reject("invalid_joint_arrays");
            if (fallback) {
                // A default command is never an observed body part. Accept
                // only our named standing reference, checked below by name.
                const auto commands = packet.at("reference_joint_mask").get<std::vector<bool>>();
                if (packet.value("reference_pose", "") != "g1_standing" || commands.size() != 29
                    || std::any_of(active.begin(), active.end(), [](bool a) { return a; })
                    || !std::all_of(commands.begin(), commands.end(), [](bool a) { return a; })
                    || !packet.at("person_id").is_null() || packet.value("base_tracked", false))
                    return reject("invalid_initial_reference");
                active = commands;
            }
            Frame frame;
            frame.q = held_; frame.active.fill(false); frame.root = held_root_;
            if (!packet.at("frame").is_number_integer()) return reject("invalid_frame_index");
            frame.index = packet.at("frame").get<int64_t>();
            // Use actual processed-frame capture intervals even for a video
            // file decoded slower than its nominal FPS.
            frame.source_time = packet.value("reference_monotonic_s",
                packet.value("capture_monotonic_s", packet.value("capture_unix_s", received)));
            if (frame.index < 0 || !std::isfinite(frame.source_time)) return reject("invalid_frame_time");
            for (int motor = 0; motor < 29; ++motor) {
                const auto it = std::find(names.begin(), names.end(), motor_names[motor]);
                if (it == names.end()) return reject("unknown_joint_names");
                const int j = int(it - names.begin());
                frame.active[motor] = active[j] && values[j].is_number();
                if (frame.active[motor]) {
                    frame.q[motor] = values[j].get<double>();
                    if (!std::isfinite(frame.q[motor])) return reject("invalid_joint_value");
                }
            }
            if (fallback) {
                if (!std::all_of(frame.active.begin(), frame.active.end(), [](bool a) { return a; })
                    || (frame.q - standing()).cwiseAbs().maxCoeff() > 1e-8)
                    return reject("invalid_initial_reference");
                frame.root = Eigen::Quaterniond::Identity();
            }
            for (int start : {0, 6}) {
                bool whole_leg = true;
                for (int i = start; i < start + 6; ++i) whole_leg &= frame.active[i];
                if (!whole_leg) for (int i = start; i < start + 6; ++i) frame.active[i] = false;
            }
            if (std::none_of(frame.active.begin(), frame.active.end(), [](bool value) { return value; })) return reject("no_active_joints");
            if (packet.value("base_tracked", false)) {
                if (packet.value("base_quaternion_order", "") != "wxyz") return reject("invalid_quaternion_order");
                const auto qpos = packet.at("qpos").get<std::vector<double>>();
                if (qpos.size() != 36) return reject("invalid_qpos_size");
                frame.root = Eigen::Quaterniond(qpos[3], qpos[4], qpos[5], qpos[6]);
                if (!frame.root.coeffs().allFinite() || frame.root.norm() < .5 || frame.root.norm() > 1.5) return reject("invalid_root_quaternion");
                frame.root.normalize();
                frame.position=Eigen::Vector3d(qpos[0],qpos[1],qpos[2]);
                if(!frame.position.allFinite())return reject("invalid_base_position");
                frame.base_tracked=true;
                for(int i=0;i<12;++i)frame.base_tracked&=frame.active[i];
            }
            if(packet.value("reference_kind",std::string("camera"))=="motion"){
                if(!frame.base_tracked||fallback)return reject("motion_requires_tracked_base_and_legs");
                const auto velocity=packet.at("base_linear_velocity_world").get<std::vector<double>>();
                if(velocity.size()!=3)return reject("invalid_motion_velocity");
                frame.linear_velocity_world=Eigen::Vector3d(velocity[0],velocity[1],velocity[2]);
                if(!frame.linear_velocity_world.allFinite())return reject("invalid_motion_velocity");
                frame.motion_reference=true;
            }
            if(packet.contains("wbc_reference")){
                const auto& anchor=packet.at("wbc_reference");
                if(!frame.motion_reference||!std::all_of(frame.active.begin(),frame.active.end(),[](bool a){return a;})
                    ||anchor.value("schema","")!="wbc-torso-v1"||anchor.value("body_name","")!="torso_link")return reject("invalid_wbc_anchor_contract");
                auto vector=[&](const char* key){const auto v=anchor.at(key).get<std::vector<double>>();
                    if(v.size()!=3)throw std::invalid_argument("invalid_wbc_anchor_vector");
                    return Eigen::Vector3d(v[0],v[1],v[2]);};
                frame.torso.position=vector("position_world");frame.torso.linear=vector("linear_velocity_world");frame.torso.angular=vector("angular_velocity_world");
                const auto quat=anchor.at("quaternion_wxyz").get<std::vector<double>>();
                if(quat.size()!=4)return reject("invalid_wbc_anchor_quaternion");
                frame.torso.orientation=Eigen::Quaterniond(quat[0],quat[1],quat[2],quat[3]);
                if(!frame.torso.position.allFinite()||!frame.torso.linear.allFinite()||!frame.torso.angular.allFinite()
                    ||!frame.torso.orientation.coeffs().allFinite()||frame.torso.orientation.norm()<.5||frame.torso.orientation.norm()>1.5)return reject("invalid_wbc_anchor_values");
                frame.torso.orientation.normalize();frame.torso.valid=true;
            }
            const std::string session = packet.value("stream_id", "legacy");
            const auto person = fallback ? person_ : packet.at("person_id");
            if (session.empty() || (!fallback && person.is_null())) return reject("missing_person_or_session");
            const bool new_session = session != session_;
            const bool after_gap = last_received_ >= 0. && received - last_received_ > timeout_;
            if (!new_session && !stopped_ && !frames_.empty() && person == person_
                && packet.value("reference_hold", false) && frame.index == last_index_
                && frame.source_time == last_source_time_ && received - last_frame_received_ <= timeout_) {
                const auto& previous = frames_.back();
                bool same = frame.active == previous.active;
                for (int i = 0; i < 29; ++i)
                    if (frame.active[i] && frame.q[i] != previous.q[i]) same = false;
                if (packet.value("base_tracked", false) || fallback)
                    same &= std::abs(frame.root.dot(previous.root)) > 1. - 1e-10;
                same &= frame.motion_reference==previous.motion_reference;
                if(frame.motion_reference)same &= (frame.linear_velocity_world-previous.linear_velocity_world).norm()<1e-10;
                same &= frame.torso.valid==previous.torso.valid;
                if(frame.torso.valid)same &= (frame.torso.position-previous.torso.position).norm()<1e-10
                    &&(frame.torso.linear-previous.torso.linear).norm()<1e-10&&(frame.torso.angular-previous.torso.angular).norm()<1e-10
                    &&std::abs(frame.torso.orientation.dot(previous.torso.orientation))>1.-1e-10;
                if (!same) return reject("changed_reference_hold");
                // Keepalive does not create artificial camera frames or
                // reduce interpolation delay to the UDP sender's period.
                last_received_ = received; ++reference_holds_; last_rejection_.clear();
                return true;
            }
            if (!new_session && (frame.index <= last_index_ || frame.source_time <= last_source_time_)) return reject("duplicate_or_out_of_order_frame");
            if (new_session || after_gap || person != person_) {
                clear_frames(); session_ = session; person_ = person;
            }
            if (!frames_.empty() && (frame.index <= frames_.back().index || frame.source_time <= frames_.back().source_time))
                return reject("duplicate_or_out_of_order_frame"); // Never refresh the watchdog.
            if (frames_.empty()) {
                origin_source_ = frame.source_time; origin_local_ = received;
            } else {
                intervals_.push_back(std::max(frame.source_time - frames_.back().source_time, received - last_frame_received_));
                if (intervals_.size() > 8) intervals_.pop_front();
            }
            frame.time = origin_local_ + frame.source_time - origin_source_;
            if (!frames_.empty()) for (int i = 0; i < 29; ++i)
                if (!frame.active[i]) frame.q[i] = frames_.back().q[i];
            frames_.push_back(frame);
            while (frames_.size() > 128) frames_.pop_front();
            last_received_ = last_frame_received_ = received; stopped_ = false;
            last_index_ = frame.index; last_source_time_ = frame.source_time;
            ++valid_poses_; last_valid_frame_ = frame.index; last_rejection_.clear();
            return true;
        } catch (const nlohmann::json::exception&) { return reject("malformed_or_missing_fields"); }
          catch (const std::invalid_argument&) { return reject("invalid_wbc_anchor_vector"); }
    }
    Reference sample(double now) {
        std::lock_guard<std::mutex> lock(mutex_);
        return sample_unlocked(now);
    }
    Window sample_window(double now) {
        // Capture one coherent generation for the entire encoder window. A
        // concurrently received person switch cannot mix two people's poses.
        std::lock_guard<std::mutex> lock(mutex_);
        Window out; out.generation = generation_; out.frames[0] = sample_unlocked(now);
        for (int i = 1; i < 10; ++i)
            out.frames[i] = out.frames[0].ready ? at(cursor_ + i * .1) : out.frames[0];
        return out;
    }
    Reference future(double seconds) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return at(cursor_ + std::max(0., seconds)); // Clamp at last received pose, no predictions.
    }
    uint64_t generation() const {
        std::lock_guard<std::mutex> lock(mutex_); return generation_;
    }
    nlohmann::json diagnostics(double now) const {
        std::lock_guard<std::mutex> lock(mutex_);
        auto age = [now](double time) -> nlohmann::json {
            return time < 0. ? nlohmann::json(nullptr) : nlohmann::json(std::max(0., now - time));
        };
        return {{"packets", packets_}, {"valid_poses", valid_poses_}, {"rejected_packets", rejected_packets_},
                {"reference_holds", reference_holds_},
                {"last_status", last_status_}, {"last_rejection", last_rejection_},
                {"last_retarget_status", last_retarget_status_}, {"last_tracking_reason", last_tracking_reason_},
                {"buffer_delay_s", delay_unlocked()},
                {"playback_lag_s", frames_.empty() || !std::isfinite(cursor_) ? nlohmann::json(nullptr)
                    : nlohmann::json(std::max(0., frames_.back().time - cursor_))},
                {"last_packet_age_s", age(last_packet_)}, {"last_valid_pose_age_s", age(last_received_)},
                {"last_pose_latency_s", last_latency_ < 0. ? nlohmann::json(nullptr) : nlohmann::json(last_latency_)},
                {"last_valid_frame", last_valid_frame_}, {"buffered_frames", frames_.size()}, {"timeout_s", timeout_}};
    }
};
} // namespace teleop

#pragma once

#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
#include <fcntl.h>
#include <chrono>
#include <stdexcept>
#include "ReferenceStream.hpp"

namespace teleop {
inline double steady_seconds() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
class UdpInput {
    int socket_ = -1;
public:
    UdpInput(const std::string& host, int port) {
        if (port < 1 || port > 65535) throw std::runtime_error("Invalid UDP port");
        socket_ = ::socket(AF_INET, SOCK_DGRAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
        if (socket_ < 0) throw std::runtime_error("Cannot create UDP receiver");
        sockaddr_in address{}; address.sin_family = AF_INET; address.sin_port = htons(port);
        if (inet_pton(AF_INET, host.c_str(), &address.sin_addr) != 1
            || bind(socket_, reinterpret_cast<sockaddr*>(&address), sizeof(address)) < 0) {
            ::close(socket_); socket_ = -1;
            throw std::runtime_error("Cannot bind UDP receiver " + host + ":" + std::to_string(port));
        }
    }
    ~UdpInput() { if (socket_ >= 0) ::close(socket_); }
    UdpInput(const UdpInput&) = delete;
    UdpInput& operator=(const UdpInput&) = delete;
    void poll(ReferenceStream& stream) {
        char buffer[65536];
        // Bound work per control tick even if a sender floods this socket.
        for (int count = 0; count < 128; ++count) {
            const auto length = recv(socket_, buffer, sizeof(buffer), 0);
            if (length < 0) {
                if (errno == EAGAIN || errno == EWOULDBLOCK) return;
                if (errno == EINTR) continue;
                throw std::runtime_error("UDP receive failed");
            }
            auto packet = nlohmann::json::parse(buffer, buffer + length, nullptr, false);
            if (!packet.is_discarded()) stream.ingest(packet, steady_seconds());
        }
    }
};
} // namespace teleop

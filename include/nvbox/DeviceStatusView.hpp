#ifndef DEVICE_STATUS_VIEW_HPP_
#define DEVICE_STATUS_VIEW_HPP_

#include <memory>
#include <iostream>

#include "IView.hpp"
#include "DeviceStatusModel.hpp"

namespace nvbox
{
    class DeviceStatusView : public IView
    {
    public:
        DeviceStatusView(const std::shared_ptr<DeviceStatusModel> model)
            : _model(model)
        {
        }

        virtual ~DeviceStatusView()
        {
        }

        void Render() const override
        {
            std::cout << std::string(30, '=') << "\n"
                      << "DeviceStatusView\n"
                      << std::string(30, '=') << "\n"
                      << std::endl;

            for (auto &&device : _model->Devices)
            {
                std::cout << "Device: " << device.UUID << "\n"
                          << std::string(30, '-') << "\n"
                          << "Temp:\t" << device.Temperature << "°C\n"
                          << std::string(30, '_') << "\n"
                          << std::endl;
            }
        }

    private:
        const std::shared_ptr<DeviceStatusModel> _model;
    };
}

#endif //! DEVICE_STATUS_VIEW_HPP_
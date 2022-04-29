#ifndef IVIEW_HPP_
#define IVIEW_HPP_

namespace nvbox
{
    class IView
    {
    protected:
        IView() {}
        virtual ~IView() {}

    public:
        virtual void Render() const = 0;
    };
}

#endif //! IVIEW_HPP_
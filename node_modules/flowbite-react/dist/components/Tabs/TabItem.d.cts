import { type ComponentProps, type FC, type ReactNode } from "react";
export interface TabItemProps extends Omit<ComponentProps<"div">, "title"> {
    active?: boolean;
    disabled?: boolean;
    icon?: FC<ComponentProps<"svg">>;
    title: ReactNode;
}
export declare const TabItem: import("react").ForwardRefExoticComponent<Omit<TabItemProps, "ref"> & import("react").RefAttributes<HTMLDivElement>>;

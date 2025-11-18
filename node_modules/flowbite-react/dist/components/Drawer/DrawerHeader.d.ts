import type { ComponentProps, FC } from "react";
import type { FlowbiteBoolean, ThemingProps } from "../../types";
export interface DrawerHeaderTheme {
    inner: {
        titleIcon: string;
        titleText: string;
        closeButton: string;
        closeIcon: string;
        titleCloseIcon: string;
    };
    collapsed: FlowbiteBoolean;
}
export interface DrawerHeaderProps extends ComponentProps<"div">, ThemingProps<DrawerHeaderTheme> {
    closeIcon?: FC<ComponentProps<"svg">>;
    title?: string;
    titleIcon?: FC<ComponentProps<"svg">>;
}
export declare const DrawerHeader: import("react").ForwardRefExoticComponent<Omit<DrawerHeaderProps, "ref"> & import("react").RefAttributes<HTMLDivElement>>;

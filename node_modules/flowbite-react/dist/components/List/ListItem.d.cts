import { type ComponentProps, type FC } from "react";
import type { ThemingProps } from "../../types";
export interface ListItemTheme {
    icon: string;
    withIcon: {
        on: string;
        off: string;
    };
}
export interface ListItemProps extends ComponentProps<"li">, ThemingProps<ListItemTheme> {
    className?: string;
    icon?: FC<ComponentProps<"svg">>;
}
export declare const ListItem: import("react").ForwardRefExoticComponent<Omit<ListItemProps, "ref"> & import("react").RefAttributes<HTMLLIElement>>;

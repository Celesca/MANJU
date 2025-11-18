import { type ComponentProps, type PropsWithChildren } from "react";
import type { ThemingProps } from "../../types";
import type { ListItemTheme } from "./ListItem";
export interface ListTheme {
    root: ListRootTheme;
    item: ListItemTheme;
}
export interface ListRootTheme {
    base: string;
    ordered: {
        on: string;
        off: string;
    };
    horizontal: string;
    unstyled: string;
    nested: string;
}
export interface ListProps extends PropsWithChildren<ComponentProps<"ul"> & ComponentProps<"ol">>, ThemingProps<ListRootTheme> {
    horizontal?: boolean;
    nested?: boolean;
    ordered?: boolean;
    unstyled?: boolean;
}
export declare const List: import("react").ForwardRefExoticComponent<Omit<ListProps, "ref"> & import("react").RefAttributes<HTMLUListElement | HTMLOListElement>>;

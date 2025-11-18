import { type ComponentProps } from "react";
import type { DynamicStringEnumKeysOf, ThemingProps } from "../../types";
import type { RatingStarSizes, RatingStarTheme } from "./RatingStar";
export interface RatingTheme {
    root: {
        base: string;
    };
    star: RatingStarTheme;
}
export interface RatingProps extends ComponentProps<"div">, ThemingProps<RatingTheme> {
    size?: DynamicStringEnumKeysOf<RatingStarSizes>;
}
export declare const Rating: import("react").ForwardRefExoticComponent<Omit<RatingProps, "ref"> & import("react").RefAttributes<HTMLDivElement>>;
